from __future__ import annotations

import asyncio
import base64
import configparser
import io
import json
from dataclasses import dataclass

import numpy as np
import os
import re
import zlib
from datetime import datetime
from io import BytesIO
from json import load
from os import listdir
from time import time
from typing import TYPE_CHECKING, Any, BinaryIO, TextIO

from PIL.ImageChops import constant
from PIL import Image

import aiohttp
import lark
from lark import Lark
from lark.lexer import Token
from lark.tree import Tree

from .. import constants, errors
from ..db import CustomLevelData, LevelData, Database
from .render import Renderer
from .variants import setup as macro_setup
from ..tile import RawTile, ReadyTile
from ..utils import cached_open

class GlobalCog:
    def __init__(self, db: Database):
        self.db = db
        self.renderer = Renderer(self.db)
        self.variant_handlers = macro_setup(self.db)
        self.reader = Reader(self.renderer, self.db)
        with open("config/leveltileoverride.json") as f:
            j = load(f)
            self.level_tile_override = j
        with open("render/tile_grammar.lark") as f:
            self.lark = Lark(f.read(), start="row", parser="lalr")

    async def handle_variant_errors(self, err: errors.VariantError):
        '''Handle errors raised in a command context by variant handlers'''
        word, variant, *rest = err.args
        msg = f"The variant `{variant}` for `{word}` is invalid"
        if isinstance(err, errors.BadTilingVariant):
            tiling = rest[0]
            return f"{msg}, since it can't be applied to tiles with tiling type `{tiling}`."
        elif isinstance(err, errors.TileNotText):
                return f"{msg}, since the tile is not text."
        elif isinstance(err, errors.BadPaletteIndex):
            return f"{msg}, since the color is outside the palette."
        elif isinstance(err, errors.BadLetterVariant):
            return f"{msg}, since letter-style text can only be 1 or 2 letters wide."
        elif isinstance(err, errors.BadMetaVariant):
            depth = rest[0]
            return f"{msg}. `{depth}` is greater than the maximum meta depth, which is `{constants.MAX_META_DEPTH}`."
        elif isinstance(err, errors.TileDoesntExist):
            return f"{msg}, since the tile doesn't exist in the database."
        elif isinstance(err, errors.UnknownVariant):
            return f"The variant `{variant}` is not valid."
        else:
            return f"{msg}."

    async def handle_custom_text_errors(self, err: errors.TextGenerationError):
        '''Handle errors raised in a command context by variant handlers'''
        text, *rest = err.args
        msg = f"The text {text} couldn't be generated automatically"
        if isinstance(err, errors.BadLetterStyle):
            return f"{msg}, since letter style can only applied to a single row of text."
        elif isinstance(err, errors.TooManyLines):
            return f"{msg}, since it has too many lines."
        elif isinstance(err, errors.LeadingTrailingLineBreaks):
            return f"{msg}, since there's `/` characters at the start or end of the text."
        elif isinstance(err, errors.BadCharacter):
            mode, char = rest
            return f"{msg}, since the letter {char} doesn't exist in '{mode}' mode."
        elif isinstance(err, errors.CustomTextTooLong):
            return f"{msg}, since it's too long ({len(text)})."
        else:
            return f"{msg}."

    async def handle_operation_errors(self, err: errors.OperationError):
        '''Handle errors raised in a command context by operation macros'''
        operation, pos, tile, *rest = err.args
        msg = f"The operation {operation} is not valid"
        if isinstance(err, errors.OperationNotFound):
            return f"The operation `{operation}` for `{tile.name}` could not be found."
        elif isinstance(err, errors.MovementOutOfFrame):
            return f"Tried to move out of bounds with the `{operation}` operation for `{tile.name}`."
        else:
            return f"The operation `{operation}` failed for `{tile.name}`."

    def parse_row(self):
        pass

    async def render_tiles(self, *, objects: str, is_rule: bool):
        '''Performs the bulk work for both `tile` and `rule` commands.'''
        start = time()
        tiles = objects.lower().strip()
        
        # replace emoji with their :text: representation
        builtin_emoji = {
            ord("\u24dc"): ":m:", # lower case circled m
            ord("\u24c2"): ":m:", # upper case circled m
            ord("\U0001f199"): ":up:", # up! emoji
            ord("\U0001f637"): ":mask:", # mask emoji
            ord("\ufe0f"): None
        }
        tiles = tiles.translate(builtin_emoji)
        tiles = re.sub(r'<a?(:[a-zA-Z0-9_]{2,32}:)\d{1,21}>', r'\1', tiles)
        
        # ignore all these
        tiles = tiles.replace("```\n", "").replace("\\", "").replace("`", "")

        # Determines if this should be a spoiler
        spoiler = tiles.count("||") >= 2
        tiles = tiles.replace("|", "")
        
        # Check for empty input
        if not tiles:
            return "Input cannot be blank."

        # Handle flags *first*, before even splitting
        flag_patterns = (
            r"(?:^|\s)(?:--background|-b)(?:=(\d)/(\d))?(?:$|\s)",
            r"(?:^|\s)(?:--palette=|-p=|palette:)(\w+)(?:$|\s)",
            r"(?:^|\s)(?:--raw|-r)(?:=([a-zA-Z_0-9]+))?(?:$|\s)",
            r"(?:^|\s)(?:--letter|-l)(?:$|\s)",
            r"(?:^|\s)(?:(--delay=|-d=)(\d+))(?:$|\s)",
            r"(?:^|\s)(?:(--frames=|-f=)(\d))(?:$|\s)",
        )
        background = None
        for match in re.finditer(flag_patterns[0], tiles):
            if match.group(1) is not None:
                tx, ty = int(match.group(1)), int(match.group(2))
                if not (0 <= tx <= 7 and 0 <= ty <= 5):
                    return "The provided background color is invalid."
                background = tx, ty
            else:
                background = (0, 4)
        palette = "default"
        for match in re.finditer(flag_patterns[1], tiles):
            palette = match.group(1)
            if palette + ".png" not in listdir("data/palettes"):
                return f"Could not find a palette with name \"{palette}\"."
        raw_output = False
        raw_name = ""
        for match in re.finditer(flag_patterns[2], tiles):
            raw_output = True
            if match.group(1) is not None:
                raw_name = match.group(1)
        default_to_letters = False
        for match in re.finditer(flag_patterns[3], tiles):
            default_to_letters = True
        delay = 200
        for match in re.finditer(flag_patterns[4], tiles):
            delay = int(match.group(2))
            if delay < 1 or delay > 1000:
                return f"Delay must be between 1 and 1000 milliseconds."
        frame_count = 3
        for match in re.finditer(flag_patterns[5], tiles):
            frame_count = int(match.group(2))
            if frame_count < 1 or frame_count > 3:
                return f"The frame count must be 1, 2 or 3."
        
        # Clean up
        for pattern in flag_patterns:
            tiles = re.sub(pattern, " ", tiles)
        
        # Split input into lines
        rows = tiles.splitlines()
        
        expanded_tiles: dict[tuple[int, int, int], list[RawTile]] = {}
        previous_tile: list[RawTile] = []
        # Do the bulk of the parsing here:
        for y, row in enumerate(rows):
            x = 0
            try:
                row_maybe = row.strip()
                if not row_maybe:
                    continue
                tree = self.lark.parse(row_maybe)
            except lark.UnexpectedCharacters as e:
                return f"Invalid character `{e.char}` in row {y}, around `... {row[e.column - 5: e.column + 5]} ...`"
            except lark.UnexpectedToken as e:
                mistake_kind = e.match_examples(
                    self.lark.parse, 
                    {
                        "unclosed": [
                            "(baba",
                            "[this ",
                            "\"rule",
                        ],
                        "missing": [
                            ":red",
                            "baba :red",
                            "&baba",
                            "baba& keke",
                            ">baba",
                            "baba> keke"
                        ],
                        "variant": [
                            "baba: keke",
                        ]
                    }
                )
                around = f"`... {row[e.column - 5 : e.column + 5]} ...`"
                if mistake_kind == "unclosed":
                    return f"Unclosed brackets or quotes! Expected them to close around {around}."
                elif mistake_kind == "missing":
                    return f"Missing a tile in row {y}! Make sure not to have spaces between `&`, `:`, or `>`!\nError occurred around {around}."
                elif mistake_kind == "variant":
                    return f"Empty variant in row {y}, around {around}."
                else:
                    return f"Invalid syntax in row {y}, around {around}."
            except lark.UnexpectedEOF as e:
                return f"Unexpected end of input in row {y}."
            for line in tree.children: 
                line: Tree
                line_text_mode: bool | None = None
                line_variants: list[str] = []

                if line.data == "text_chain":
                    line_text_mode = True
                elif line.data == "tile_chain":
                    line_text_mode = False
                elif line.data == "text_block":
                    line_text_mode = True
                elif line.data == "tile_block":
                    line_text_mode = False
                
                if line.data in ("text_block", "tile_block", "any_block"):
                    *stacks, variants = line.children 
                    variants: Tree
                    for variant in variants.children: 
                        variant: Token
                        line_variants.append(variant.value)
                else:
                    stacks = line.children
                
                for stack in stacks: 
                    stack: Tree

                    blobs: list[tuple[bool | None, list[str], Tree]] = []

                    if stack.data == "blob_stack":
                        for variant_blob in stack.children:
                            blob, variants = variant_blob.children 
                            blob: Tree
                            variants: Tree
                            
                            blob_text_mode: bool | None = None
                            
                            stack_variants = []
                            for variant in variants.children:
                                variant: Token
                                stack_variants.append(variant.value)
                            if blob.data == "text_blob":
                                blob_text_mode = True
                            elif blob.data == "tile_blob":
                                blob_text_mode = False
                            
                            blobs.append((blob_text_mode, stack_variants, blob))
                    else:
                        blobs = [(None, [], stack)] 

                    for blob_text_mode, stack_variants, blob in blobs:
                        for process in blob.children: 
                            process: Tree
                            t = 0

                            unit, *changes = process.children 
                            unit: Tree
                            changes: list[Tree]
                            
                            object, variants = unit.children 
                            object: Token
                            obj = object.value
                            variants: Tree
                            
                            final_variants: list[str] = [
                                var.value 
                                for var in variants.children
                            ]

                            def append_extra_variants(final_variants: list[str]):
                                '''IN PLACE'''
                                final_variants.extend(stack_variants)
                                final_variants.extend(line_variants)

                            def handle_text_mode(obj: str) -> str:
                                '''RETURNS COPY'''
                                text_delta = -1 if blob_text_mode is False else blob_text_mode or 0
                                text_delta += -1 if line_text_mode is False else line_text_mode or 0
                                text_delta += is_rule
                                if text_delta == 0:
                                    return obj
                                elif text_delta > 0:
                                    for _ in range(text_delta):
                                        if obj.startswith("tile_"):
                                            obj = obj[5:]
                                        else:
                                            obj = f"text_{obj}"
                                    return obj
                                else:
                                    for _ in range(text_delta):
                                        if obj.startswith("text_"):
                                            obj = obj[5:]
                                        else:
                                            raise RuntimeError("this should never happen")
                                            # TODO: handle this explicitly
                                    return obj

                            obj = handle_text_mode(obj)
                            append_extra_variants(final_variants)

                            dx = dy = 0
                            temp_tile: list[RawTile] = [RawTile(obj, final_variants, ephemeral=False)]
                            last_hack = False
                            for change in changes:
                                if change.data == "transform":
                                    last_hack = False
                                    seq, unit = change.children 
                                    seq: str

                                    count = len(seq)
                                    still = temp_tile.pop()
                                    still.ephemeral = True
                                    if still.is_previous:
                                        still = previous_tile[-1]
                                    else:
                                        previous_tile[-1:] = [still]
                                    
                                    for dt in range(count):
                                        expanded_tiles.setdefault((x + dx, y + dy, t + dt), []).append(still)
                                        
                                    object, variants = unit.children 
                                    object: Token
                                    obj = object.value
                                    obj = handle_text_mode(obj)
                                    
                                    final_variants = [var.value for var in variants.children]
                                    append_extra_variants(final_variants)
                                    
                                    temp_tile.append(
                                        RawTile(
                                            obj,
                                            final_variants,
                                            ephemeral=False
                                        )
                                    )
                                    t += count

                                elif change.data == "operation":
                                    last_hack = True
                                    oper = change.children[0] 
                                    oper: Token
                                    try:
                                        ddx, ddy, dt = self.bot.operation_macros.expand_into(
                                            expanded_tiles,
                                            temp_tile,
                                            (x + dx, y + dy, t),
                                            oper.value
                                        )
                                    except errors.OperationError as err:
                                        return await self.handle_operation_errors(err)
                                    dx += ddx
                                    dy += ddy
                                    t += dt
                            # somewhat monadic behavior
                            if not last_hack:
                                expanded_tiles.setdefault((x + dx, y + dy, t), []).extend(temp_tile[:])
                    x += 1

        # Get the dimensions of the grid
        width = max(expanded_tiles, key=lambda pos: pos[0])[0] + 1
        height = max(expanded_tiles, key=lambda pos: pos[1])[1] + 1
        duration = 1 + max(t for _, _, t in expanded_tiles)

        temporal_maxima: dict[tuple[int, int], tuple[int, list[RawTile]]] = {}
        for (x, y, t), tile_stack in expanded_tiles.items():
            if (x, y) in temporal_maxima and temporal_maxima[x, y][0] < t:
                persistent = [tile for tile in tile_stack if not tile.ephemeral]
                if len(persistent) != 0:
                    temporal_maxima[x, y] = t, persistent
            elif (x, y) not in temporal_maxima:
                persistent = [tile for tile in tile_stack if not tile.ephemeral]
                if len(persistent) != 0:
                    temporal_maxima[x, y] = t, persistent
        # Pad the grid across time
        for (x, y), (t_star, tile_stack) in temporal_maxima.items():
            for t in range(t_star, duration - 1):
                if (x, y, t + 1) not in expanded_tiles:
                    expanded_tiles[x, y, t + 1] = tile_stack
                else:
                    expanded_tiles[x, y, t + 1] = tile_stack + expanded_tiles[x, y, t + 1]
                    
        # filter out blanks before rendering
        expanded_tiles = {index: [tile for tile in stack if not tile.is_empty] for index, stack in expanded_tiles.items()}
        expanded_tiles = {index: stack for index, stack in expanded_tiles.items() if len(stack) != 0}

        # Don't proceed if the request is too large.
        # (It shouldn't be that long to begin with because of Discord's 2000 character limit)
        if width * height * duration > constants.MAX_VOLUME:
            return f"Too large of an animation ({width * height * duration}). An animation may have up to {constants.MAX_VOLUME} tiles, including tiles repeated in frames."
        if width > constants.MAX_WIDTH:
            return f"Too wide ({width}). You may only render scenes up to {constants.MAX_WIDTH} tiles wide."
        if height > constants.MAX_HEIGHT:
            return f"Too high ({height}). You may only render scenes up to {constants.MAX_HEIGHT} tiles tall."
        if duration > constants.MAX_DURATION:
            return f"Too many frames ({duration}). You may only render scenes with up to {constants.MAX_DURATION} animation frames."
        
        try:
            # Handles variants based on `:` affixes
            buffer = BytesIO()
            extra_buffer = BytesIO() if raw_output else None
            extra_names = [] if raw_output else None
            full_objects = await self.variant_handlers.handle_grid(
                expanded_tiles,
                (width, height),
                raw_output=raw_output,
                extra_names=extra_names,
                default_to_letters=default_to_letters
            )
            if extra_names is not None and not raw_name:
                if len(extra_names) == 1:
                    raw_name = extra_names[0]
                else:
                    raw_name = constants.DEFAULT_RENDER_ZIP_NAME
            full_tiles = await self.renderer.render_full_tiles(
                full_objects,
                palette=palette,
                random_animations=True
            )
            await self.renderer.render(
                full_tiles,
                grid_size=(width, height),
                duration=duration,
                palette=palette,
                background=background, 
                out=buffer,
                delay=delay,
                frame_count=frame_count,
                upscale=not raw_output,
                extra_out=extra_buffer,
                extra_name=raw_name,
            )
        except errors.TileNotFound as e:
            word = e.args[0]
            name = word.name
            if word.name.startswith("tile_") and await self.db.tile(name[5:]) is not None:
                return f"The tile `{name}` could not be found. Perhaps you meant `{name[5:]}`?"
            if await self.db.tile("text_" + name) is not None:
                return f"The tile `{name}` could not be found. Perhaps you meant `{'text_' + name}`?"
            return f"The tile `{name}` could not be found."
        except errors.BadTileProperty as e:
            word, (w, h) = e.args
            return f"The tile `{word.name}` could not be made into a property, because it's too big (`{w} by {h}`)."
        except errors.EmptyTile as e:
            return "Cannot use blank tiles in that context."
        except errors.EmptyVariant as e:
            word = e.args[0]
            return f"You provided an empty variant for `{word.name}`."
        except errors.VariantError as e:
            return await self.handle_variant_errors(e)
        except errors.TextGenerationError as e:
            return await self.handle_custom_text_errors(e)
        
        filename = datetime.utcnow().strftime(r"render_%Y-%m-%d_%H.%M.%S.gif")
        delta = time() - start
        msg = f"*Rendered in {delta:.2f} s*"
        if extra_buffer is not None and raw_name:
            extra_buffer.seek(0)
            return buffer, extra_buffer
        else:
            return buffer

    async def search_levels(self, query: str, **flags: Any) -> list[tuple[tuple[str, str], LevelData]]:
        '''Finds levels by query.
        
        Flags:
        * `map`: Which map screen the level is from.
        * `world`: Which levelpack / world the level is from.
        '''
        levels: list[tuple[tuple[str, str], LevelData]] = []
        found: set[tuple[str, str]] = set()
        f_map = flags.get("map")
        f_world = flags.get("world")
        async with self.db.conn.cursor() as cur:
            # [world]/[levelid]
            parts = query.split("/", 1)
            if len(parts) == 2:
                await cur.execute(
                    '''
                    SELECT * FROM levels 
                    WHERE 
                        world == :world AND
                        id == :id AND (
                            :f_map IS NULL OR LOWER(parent) == LOWER(:f_map)
                        ) AND (
                            :f_world IS NULL OR LOWER(world) == LOWER(:f_world)
                        );
                    ''',
                    dict(world=parts[0], id=parts[1], f_map=f_map, f_world=f_world)
                )
                row = await cur.fetchone()
                if row is not None:
                    data = LevelData.from_row(row)
                    if (data.world, data.id) not in found:
                        found.add((data.world, data.id))
                        levels.append(((data.world, data.id), data))


            # This system ensures that baba worlds are 
            # *always* prioritized over modded worlds,
            # even if the modded query belongs to a higher tier.
            # 
            # A real example of the naive approach failing is 
            # with the query "map", matching `baba/106level` by name
            # and `alphababa/map` by level ID. Even though name 
            # matches are lower priority than ID matches, we want
            # ths to return `baba/106level` first.
            maybe_parts = query.split(" ", 1)
            if len(maybe_parts) == 2:
                possible_queries = [
                    ("baba", query),
                    (maybe_parts[0], maybe_parts[1]),
                    (f_world, query)
                ]
            else:
                possible_queries = [
                    ("baba", query),
                    (f_world, query)
                ]
            if f_world is not None:
                possible_queries = possible_queries[1:]

            for f_world, query in possible_queries:
                # someworld/[levelid]
                await cur.execute(
                    '''
                    SELECT * FROM levels
                    WHERE id == :id AND (
                        :f_map IS NULL OR LOWER(parent) == LOWER(:f_map)
                    ) AND (
                        :f_world IS NULL OR LOWER(world) == LOWER(:f_world)
                    )
                    ORDER BY CASE world 
                        WHEN :default
                        THEN NULL 
                        WHEN :museum
                        THEN ""
                        WHEN :new_adv
                        THEN ""
                        ELSE world 
                    END ASC;
                    ''',
                    dict(
                        id=query, 
                        f_map=f_map, 
                        f_world=f_world, 
                        default=constants.BABA_WORLD, 
                        museum=constants.MUSEUM_WORLD, 
                        new_adv=constants.NEW_ADVENTURES_WORLD
                    )
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    if (data.world, data.id) not in found:
                        found.add((data.world, data.id))
                        levels.append(((data.world, data.id), data))
                
                # [parent]-[map_id]
                segments = query.split("-")
                if len(segments) == 2:
                    await cur.execute(
                        '''
                        SELECT * FROM levels 
                        WHERE LOWER(parent) == LOWER(:parent) AND (
                            UNLIKELY(map_id == :map_id) OR (
                                style == 0 AND 
                                CAST(number AS TEXT) == :map_id
                            ) OR (
                                style == 1 AND
                                LENGTH(:map_id) == 1 AND
                                number == UNICODE(:map_id) - UNICODE("a")
                            ) OR (
                                style == 2 AND 
                                SUBSTR(:map_id, 1, 5) == "extra" AND
                                number == CAST(TRIM(SUBSTR(:map_id, 6)) AS INTEGER) - 1
                            )
                        ) AND (
                            :f_map IS NULL OR LOWER(parent) == LOWER(:f_map)
                        ) AND (
                            :f_world IS NULL OR LOWER(world) == LOWER(:f_world)
                        ) ORDER BY CASE world 
                            WHEN :default
                            THEN NULL 
                            WHEN :museum
                            THEN ""
                            WHEN :new_adv
                            THEN ""
                            ELSE world 
                        END ASC;
                        ''',
                        dict(
                            parent=segments[0], 
                            map_id=segments[1], 
                            f_map=f_map, 
                            f_world=f_world, 
                            default=constants.BABA_WORLD,
                            museum=constants.MUSEUM_WORLD, 
                            new_adv=constants.NEW_ADVENTURES_WORLD
                        )
                    )
                    for row in await cur.fetchall():
                        data = LevelData.from_row(row)
                        if (data.world, data.id) not in found:
                            found.add((data.world, data.id))
                            levels.append(((data.world, data.id), data))

                # [name]
                await cur.execute(
                    '''
                    SELECT * FROM levels
                    WHERE name == :name AND (
                        :f_map IS NULL OR LOWER(parent) == LOWER(:f_map)
                    ) AND (
                        :f_world IS NULL OR LOWER(world) == LOWER(:f_world)
                    )
                    ORDER BY CASE world 
                        WHEN :default
                        THEN NULL 
                        WHEN :museum
                        THEN ""
                        WHEN :new_adv
                        THEN ""
                        ELSE world 
                    END ASC;
                    ''',
                    dict(
                        name=query, 
                        f_map=f_map, 
                        f_world=f_world, 
                        default=constants.BABA_WORLD, 
                        museum=constants.MUSEUM_WORLD, 
                        new_adv=constants.NEW_ADVENTURES_WORLD
                    )
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    if (data.world, data.id) not in found:
                        found.add((data.world, data.id))
                        levels.append(((data.world, data.id), data))

                # [name-ish]
                await cur.execute(
                    '''
                    SELECT * FROM levels
                    WHERE INSTR(name, :name) AND (
                        :f_map IS NULL OR LOWER(parent) == LOWER(:f_map)
                    ) AND (
                        :f_world IS NULL OR LOWER(world) == LOWER(:f_world)
                    )
                    ORDER BY COALESCE(
                        CASE world 
                            WHEN :default
                            THEN NULL 
                            WHEN :museum
                            THEN ""
                            WHEN :new_adv
                            THEN ""
                            ELSE world 
                        END,
                        INSTR(name, :name)
                    ) ASC, number DESC;
                    ''',
                    dict(
                        name=query, 
                        f_map=f_map, 
                        f_world=f_world, 
                        default=constants.BABA_WORLD, 
                        museum=constants.MUSEUM_WORLD, 
                        new_adv=constants.NEW_ADVENTURES_WORLD
                    )
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    if (data.world, data.id) not in found:
                        found.add((data.world, data.id))
                        levels.append(((data.world, data.id), data))

                # [map_id]
                await cur.execute(
                    '''
                    SELECT * FROM levels 
                    WHERE LOWER(map_id) == LOWER(:map) AND parent IS NULL AND (
                        :f_map IS NULL OR LOWER(map_id) == LOWER(:f_map)
                    ) AND (
                        :f_world IS NULL OR LOWER(world) == LOWER(:f_world)
                    )
                    ORDER BY CASE world 
                        WHEN :default
                        THEN NULL
                        ELSE world
                    END ASC;
                    ''',
                    dict(map=query, f_map=f_map, f_world=f_world, default=constants.BABA_WORLD)
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    if (data.world, data.id) not in found:
                        found.add((data.world, data.id))
                        levels.append(((data.world, data.id), data))
        
        return levels

    async def level_command(self, *, query: str):
        '''Renders the Baba Is You level from a search term.

        Levels are searched for in the following order:
        * Custom level code (e.g. "1234-ABCD")
        * World & level ID (e.g. "baba/20level")
        * Level ID (e.g. "16level")
        * Level number (e.g. "space-3" or "lake-extra 1")
        * Level name (e.g. "further fields")
        * The map ID of a world (e.g. "cavern", or "lake")
        '''
        return await self.perform_level_command(query, mobile=False)

    async def mobile(self, *, query: str):
        '''Renders the mobile Baba Is You level from a search term.

        Levels are searched for in the following order:
        * World & level ID (e.g. "baba/20level")
        * Level ID (e.g. "16level")
        * Level number (e.g. "space-3" or "lake-extra 1")
        * Level name (e.g. "further fields")
        * The map ID of a world (e.g. "cavern", or "lake")
        '''
        return await self.perform_level_command(query, mobile=True)
    
    async def perform_level_command(self, query: str, *, mobile: bool):
        custom_level: CustomLevelData | None = None
        
        spoiler = query.count("||") >= 2
        fine_query = query.lower().strip().replace("|", "")
        
        # [abcd-0123]
        if re.match(r"^[a-z0-9]{4}\-[a-z0-9]{4}$", fine_query) and not mobile:
            row = await self.db.conn.fetchone(
                '''
                SELECT * FROM custom_levels WHERE code == ?;
                ''',
                fine_query
            )
            if row is not None:
                custom_level = CustomLevelData.from_row(row)
            else:
                async with aiohttp.request("GET", f"https://baba-is-bookmark.herokuapp.com/api/level/exists?code={fine_query.upper()}") as resp:
                    if resp.status in (200, 304):
                        data = await resp.json()
                        if data["data"]["exists"]:
                            try:
                                custom_level = await self.reader.render_custom_level(fine_query)
                            except ValueError as e:
                                size = e.args[0]
                                return f"The level code is valid, but the level's width, height or area is way too big ({size})!"
                            except aiohttp.ClientResponseError as e:
                                return f"The Baba Is Bookmark site returned a bad response. Try again later."
        if custom_level is None:
            levels = await self.search_levels(fine_query)
            if len(levels) == 0:
                return "A level could not be found with that query."
            _, level = levels[0]
        else:
            levels = {}
            level = custom_level

        if isinstance(level, LevelData):
            path = level.unique()
            display = level.display()
            rows = [
                f"Name: ||`{display}`||" if spoiler else f"Name: `{display}`",
                f"ID: `{path}`",
            ]
            if level.subtitle:
                rows.append(
                    f"Subtitle: `{level.subtitle}`"
                )
            mobile_exists = os.path.exists(f"target/renders/{level.world}_m/{level.id}.gif")
            
            if not mobile and mobile_exists:
                rows.append(
                    f"*This level is also on mobile, see `+level mobile {level.unique()}`*"
                )
            elif mobile and mobile_exists:
                rows.append(
                    f"*This is the mobile version. For others, see `+level {level.unique()}`*"
                )

            if mobile and mobile_exists:
                gif = f"target/renders/{level.world}_m/{level.id}.gif"
            elif mobile and not mobile_exists:
                rows.append("*This level doesn't have a mobile version. Using the normal gif instead...*")
                gif = f"target/renders/{level.world}/{level.id}.gif"
            else:
                gif = f"target/renders/{level.world}/{level.id}.gif"
        else:
            gif = f"target/renders/levels/{level.code}.gif"
            path = level.unique()
            display = level.name
            rows = [
                f"Name: ||`{display}`|| (by `{level.author}`)" 
                    if spoiler else f"Name: `{display}` (by `{level.author}`)",
                f"Level code: `{path}`",
            ]
            if level.subtitle:
                rows.append(
                    f"Subtitle: `{level.subtitle}`"
                )

        if len(levels) > 1:
            levels = levels[1:]
            extras = [level.unique() for _, level in levels]
            if len(levels) > constants.OTHER_LEVELS_CUTOFF:
                extras = extras[:constants.OTHER_LEVELS_CUTOFF]
            paths = ", ".join(f"`{extra}`" for extra in extras)
            plural = "result" if len(extras) == 1 else "results"
            suffix = ", `...`" if len(levels) > constants.OTHER_LEVELS_CUTOFF else ""
            rows.append(
                f"*Found {len(levels)} other {plural}: {paths}{suffix}*"
            )

        formatted = "\n".join(rows)

        return BytesIO(open(gif, "rb").read()), formatted

def flatten(x: int, y: int, width: int) -> int:
    '''Return the flattened position of a coordinate in a grid of specified width'''
    return int(y) * width + int(x)

class Grid:
    '''This stores the information of a single Baba level, in a format readable by the renderer.'''

    def __init__(self, filename: str, world: str):
        '''Initializes a blank grid, given a path to the level file.
        This should not be used; you should use Reader.read_map() instead to generate a filled grid.
        '''
        # The location of the level
        self.fp: str = f"data/levels/{world}/{filename}.l"
        self.filename: str = filename
        self.world: str = world
        # Basic level information
        self.name: str = ""
        self.subtitle: str | None = None
        self.palette: str = "default"
        self.images: list[str] = []
        # Object information
        self.width: int = 0
        self.height: int = 0
        self.cells: list[list[Item]] = []
        # Parent level and map identification
        self.parent: str | None = None
        self.map_id: str | None = None
        self.style: int | None = None
        self.number: int | None = None
        # Custom levels
        self.author: str | None = None

    def ready_grid(self, *, remove_borders: bool):
        '''Returns a ready-to-paste version of the grid.'''

        def is_adjacent(sprite: str, x: int, y: int) -> bool:
            valid = (sprite, "edge", "level")
            if x == 0 or x == self.width - 1:
                return True
            if y == 0 or y == self.height - 1:
                return True
            return any(item.sprite in valid for item in self.cells[y * self.width + x])

        def open_sprite(world: str, sprite: str, variant: int, wobble: int, *,
                        cache: dict[str, Image.Image]) -> Image.Image:
            '''This first checks the given world, then the `baba` world, then `baba-extensions`, and if both fail it returns `default`'''
            if sprite == "icon":
                path = f"data/sprites/{{}}/icon.png"
            elif sprite.startswith("icon_default"):
                path = f"data/sprites/{{}}/{sprite}.png"
            elif sprite in ("smiley", "hi") or sprite.startswith("icon"):
                path = f"data/sprites/{{}}/{sprite}_1.png"
            elif sprite == "default":
                path = f"data/sprites/{{}}/default_{wobble}.png"
            else:
                path = f"data/sprites/{{}}/{sprite}_{variant}_{wobble}.png"

            for maybe_world in (world, constants.BABA_WORLD, constants.EXTENSIONS_WORLD):
                try:
                    return cached_open(path.format(maybe_world), cache=cache, fn=Image.open).convert("RGBA")
                except FileNotFoundError:
                    continue
            else:
                return cached_open(f"data/sprites/{constants.BABA_WORLD}/default_{wobble}.png", cache=cache,
                                   fn=Image.open).convert("RGBA")

        def recolor(sprite: Image.Image, rgb: tuple[int, int, int]) -> Image.Image:
            '''Apply rgb color multiplication (0-255)'''
            r, g, b = rgb
            arr = np.asarray(sprite, dtype='float64')
            arr[..., 0] *= r / 256
            arr[..., 1] *= g / 256
            arr[..., 2] *= b / 256
            return Image.fromarray(arr.astype('uint8'))

        sprite_cache = {}
        grid = {}
        palette_img = Image.open(f"data/palettes/{self.palette}.png").convert("RGB")
        for y in range(self.height):
            for x in range(self.width):
                if remove_borders and (x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1):
                    continue
                for item in sorted(self.cells[y * self.width + x], key=lambda item: item.layer):
                    item: Item
                    if item.tiling in constants.DIRECTION_TILINGS:
                        variant = item.direction * 8
                    elif item.tiling in constants.AUTO_TILINGS:
                        variant = (
                                is_adjacent(item.sprite, x + 1, y) * 1 +
                                is_adjacent(item.sprite, x, y - 1) * 2 +
                                is_adjacent(item.sprite, x - 1, y) * 4 +
                                is_adjacent(item.sprite, x, y + 1) * 8
                        )
                    else:
                        variant = 0
                    color = palette_img.getpixel(item.color)
                    frames = (
                        recolor(open_sprite(self.world, item.sprite, variant, 1, cache=sprite_cache), color),
                        recolor(open_sprite(self.world, item.sprite, variant, 2, cache=sprite_cache), color),
                        recolor(open_sprite(self.world, item.sprite, variant, 3, cache=sprite_cache), color),
                    )
                    grid.setdefault((x - 1, y - 1, 0), []).append(ReadyTile(frames))

        return grid

@dataclass
class Item:
    '''Represents an object within a level with metadata.
    This may be a regular object, a path object, a level object, a special object or empty.
    '''
    id: int
    layer: int
    obj: str
    sprite: str = "error"
    color: tuple[int, int] = (0, 3)
    direction: int = 0
    tiling: int = -1

    def copy(self):
        return Item(id=self.id, obj=self.obj, sprite=self.sprite, color=self.color, direction=self.direction,
                    layer=self.layer, tiling=self.tiling)

    @classmethod
    def edge(cls) -> Item:
        '''Returns an Item representing an edge tile.'''
        return cls(id=0, obj="edge", sprite="edge", layer=20)

    @classmethod
    def empty(cls) -> Item:
        '''Returns an Item representing an empty tile.'''
        return cls(id=-1, obj="empty", sprite="empty", layer=0)

    @classmethod
    def level(cls, color: tuple[int, int] = (0, 3)) -> Item:
        '''Returns an Item representing a level object.'''
        return cls(id=-2, obj="level", sprite="level", color=color, layer=20)

    @classmethod
    def icon(cls, sprite: str) -> Item:
        '''Level icon'''
        if sprite == "icon":
            sprite = sprite
        elif sprite.startswith("icon_default"):
            sprite = sprite
        elif sprite.startswith("icon"):
            sprite = sprite[:-2]  # strip _1 for icon sprites
        else:
            sprite = sprite[:-4]  # strip _0_2 for normal sprites
        return cls(id=-3, obj="icon", sprite=sprite, layer=30)


class Reader:
    '''A class for parsing the contents of level files.'''

    def __init__(self, renderer: Renderer, db: Database):
        '''Initializes the Reader cog.
        Populates the default objects cache from a data/values.lua file.
        '''
        self.renderer = renderer
        self.db = db
        self.defaults_by_id: dict[int, Item] = {}
        self.defaults_by_object: dict[str, Item] = {}
        self.defaults_by_name: dict[str, Item] = {}
        self.parent_levels: dict[str, tuple[str, dict[str, tuple[int, int]]]] = {}

        self.read_objects()

    async def render_custom_level(self, code: str) -> CustomLevelData:
        '''Renders a custom level. code should be valid (but is checked regardless)'''
        async with aiohttp.request("GET",
                                   f"https://baba-is-bookmark.herokuapp.com/api/level/raw/l?code={code.upper()}") as resp:
            resp.raise_for_status()
            data = await resp.json()
            b64 = data["data"]
            decoded = base64.b64decode(b64)
            raw_l = io.BytesIO(decoded)
        async with aiohttp.request("GET",
                                   f"https://baba-is-bookmark.herokuapp.com/api/level/raw/ld?code={code.upper()}") as resp:
            resp.raise_for_status()
            data = await resp.json()
            raw_s = data["data"]
            raw_ld = io.StringIO(raw_s)

        grid = self.read_map(code, source="levels", data=raw_l)
        grid = await self.read_metadata(grid, data=raw_ld, custom=True)

        objects = grid.ready_grid(remove_borders=True)
        out = f"target/renders/levels/{code}.gif"
        await self.renderer.render(
            objects,
            grid_size=(grid.width - 2, grid.height - 2),
            duration=1,
            palette=grid.palette,
            background=(0, 4),
            out=out
        )

        data = CustomLevelData(code.lower(), grid.name, grid.subtitle, grid.author)

        await self.db.conn.execute(
            '''
            INSERT INTO custom_levels
            VALUES (?, ?, ?, ?)
            ON CONFLICT(code) 
            DO NOTHING;
            ''',
            code.lower(), grid.name, grid.subtitle, grid.author
        )

        return data

    async def render_level(
            self,
            filename: str,
            source: str,
            initialize: bool = False,
            remove_borders: bool = False,
            keep_background: bool = False,
    ) -> LevelData:
        '''Loads and renders a level, given its file path and source.
        Shaves off the borders if specified.
        '''
        # Data
        grid = self.read_map(filename, source=source)
        grid = await self.read_metadata(grid, initialize_level_tree=initialize)
        objects = grid.ready_grid(remove_borders=remove_borders)

        # (0,4) is the color index for level backgrounds
        background = (0, 4) if keep_background else None

        # Render the level
        await self.renderer.render(
            objects,
            grid_size=(grid.width - 2 * remove_borders, grid.height - 2 * remove_borders),
            duration=1,
            palette=grid.palette,
            images=grid.images,
            image_source=grid.world,
            background=background,
            out=f"target/renders/{grid.world}/{grid.filename}.gif",
        )
        # Return level metadata
        return LevelData(filename, source, grid.name, grid.subtitle, grid.number, grid.style, grid.parent,
                         grid.map_id)

    async def load_map(self, source: str, filename: str):
        '''Loads a given level's image.'''
        # Parse and render
        await self.render_level(
            filename,
            source=source,
            initialize=False,
            remove_borders=True,
            keep_background=True,
        )

    async def clean_metadata(self, metadata: dict[str, LevelData]):
        '''Cleans up level metadata from `self.parent_levels` as well as the given dict, and updates the DB.'''

        for map_id, child_levels in self.parent_levels.values():
            remove = []
            for child_id in child_levels:
                # remove levels which point to maps themselves (i.e. don't mark map as "lake-blah: map")
                # as a result of this, every map will have no parent in its name - so it'll just be
                # something like "chasm" or "center"
                if self.parent_levels.get(child_id) is not None:
                    remove.append(child_id)
            # avoid mutating a dict while iterating over it
            for child_id in remove:
                child_levels.pop(child_id)
        for map_id, child_levels in self.parent_levels.values():
            for child_id, (number, style) in child_levels.items():
                try:
                    metadata[child_id].parent = map_id
                    metadata[child_id].number = number
                    metadata[child_id].style = style
                except KeyError:
                    # something wrong with the levelpack,
                    # nonexistent things specified
                    pass

        self.parent_levels.clear()
        await self.db.conn.executemany(
            '''
            INSERT INTO levels VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id, world) DO UPDATE SET
                name=excluded.name,
                subtitle=excluded.subtitle,
                number=excluded.number,
                style=excluded.style,
                parent=excluded.parent,
                map_id=excluded.map_id;
            ''',
            [(l.id, l.world, l.name.lower(), l.subtitle, l.number, l.style, l.parent, l.map_id) for l in
             metadata.values()]
        )

    async def load_world(self, world: str = constants.BABA_WORLD, also_mobile: bool = True):
        '''Loads and renders levels in a world and its mobile variant.
        Initializes the level tree unless otherwise specified.
        Cuts off borders from rendered levels unless otherwise specified.
        '''
        total = await self.load_single_world(world, also_mobile=also_mobile)

    async def load_all_worlds(self):
        '''NUCLEAR COMMAND!
        PLEASE DON'T USE IT UNLESS YOU NEED TO RE-RENDER EVERY PACK.
        '''
        with open("data/levelpacks.json") as f:
            packs = json.load(f)
        pack_names = ["baba"] + list(packs.keys())
        for world in pack_names:
            total = await self.load_single_world(world, also_mobile=True)

    async def load_single_world(self, world: str, *, also_mobile: bool) -> int:
        # Parse and render the level map
        levels = [l[:-2] for l in listdir(f"data/levels/{world}") if l.endswith(".l")]
        try:
            os.mkdir(f"target/renders/{world}")
        except FileExistsError:
            pass
        metadatas = {}
        total = len(levels)
        for i, level in enumerate(levels):
            metadata = await self.render_level(
                level,
                source=world,
                initialize=True,
                remove_borders=True,
                keep_background=True,
            )
            if also_mobile:
                try:
                    await self.render_level(
                        level,
                        source=f"{world}_m",
                        initialize=False,
                        remove_borders=True,
                        keep_background=True,
                    )
                except FileNotFoundError:
                    pass
            metadatas[level] = metadata
        await self.clean_metadata(metadatas)
        return total

    def read_objects(self) -> None:
        '''Inner function that parses the contents of the data/values.lua file.
        '''
        with open("data/values.lua", errors="replace") as fp:
            data = fp.read()

        start = data.find("tileslist =\n")
        end = data.find("\n}\n", start)

        assert start > 0 and end > 0
        spanned = data[start:end]

        object_pattern = re.compile(
            r"(object\d+) =\n\t\{"
            r"\n.*"
            r"\n\s*sprite = \"([^\"]*)\","
            r"\n.*\n.*\n\s*tiling = (-1|\d),"
            r"\n.*"
            r"\n\s*(?:argextra = .*,\n\s*)?(?:argtype = .*,\n\s*)?"
            r"colour = \{(\d), (\d)\},"
            r"(?:\n\s*active = \{(\d), (\d)\},)?"
            r"\n\s*tile = \{(\d+), (\d+)\},"
            r"\n.*"
            r"\n\s*layer = (\d+),"
            r"\n\s*\}",
        )
        for match in re.finditer(object_pattern, spanned):
            obj, sprite, tiling, c_x, c_y, a_x, a_y, t_x, t_y, layer = match.groups()
            if a_x is None or a_y is None:
                color = int(c_x), int(c_y)
            else:
                color = int(a_x), int(a_y)
            item = Item(
                obj=obj,
                layer=int(layer),
                id=(int(t_y) << 8) | int(t_x),
                sprite=sprite,
                tiling=int(tiling),
                color=color
            )
            self.defaults_by_id[item.id] = item
            self.defaults_by_object[obj] = item
            self.defaults_by_name[item.sprite] = item
        # We've parsed and stored all objects from data/values.lua in cache.
        # Now we only need to add the special cases:
        # Empty tiles
        empty = Item.empty()
        self.defaults_by_object[empty.obj] = empty
        self.defaults_by_id[empty.id] = empty
        self.defaults_by_name[empty.sprite] = empty
        # Level tiles
        level = Item.level()
        self.defaults_by_object[level.obj] = level
        self.defaults_by_id[level.id] = level
        self.defaults_by_name[level.sprite] = level

    def read_map(self, filename: str, source: str, data: BinaryIO | None = None) -> Grid:
        '''Parses a .l file's content, given its file path.
        Returns a Grid object containing the level data.
        '''
        grid = Grid(filename, source)
        if data is None:
            stream = open(grid.fp, "rb")
        else:
            stream = data
        stream.read(28)  # don't care about these headers
        buffer = stream.read(2)
        layer_count = int.from_bytes(buffer, byteorder="little")
        # version is assumed to be 261 (it is for all levels as far as I can tell)
        for _ in range(layer_count):
            self.read_layer(stream, grid)
        return grid

    async def read_metadata(self, grid: Grid, initialize_level_tree: bool = False, data: TextIO | None = None,
                            custom: bool = False) -> Grid:
        '''Add everything that's not just basic tile positions & IDs'''
        # We've added the basic objects & their directions.
        # Now we add everything else:
        if data is None:
            fp = open(grid.fp + "d", errors="replace", encoding="utf-8")
        else:
            fp = data

        # Strict mode must be disabled to match
        # the game's lenient parsing
        config = configparser.ConfigParser(strict=False)
        config.read_file(fp)

        # Name and palette should never be missing, but I can't guarantee this for custom levels
        grid.name = config.get("general", "name", fallback="name missing")
        grid.palette = config.get("general", "palette", fallback="default.png")[:-4]  # strip .png
        grid.subtitle = config.get("general", "subtitle", fallback=None)
        grid.map_id = config.get("general", "mapid", fallback=None)

        if custom:
            # difficulty_string = config.get("general", "difficulty", fallback=None)
            grid.author = config.get("general", "author", fallback=None)

        # Only applicable to old style cursors
        # "cursor not visible" is denoted with X and Y set to -1
        cursor_x = config.getint("general", "selectorX", fallback=-1)
        cursor_y = config.getint("general", "selectorY", fallback=-1)
        if cursor_y != -1 and cursor_x != -1:
            cursor = self.defaults_by_name["cursor"]
            pos = flatten(cursor_x, cursor_y, grid.width)
            grid.cells[pos].append(cursor)

        # Add path objects to the grid (they're not in the normal objects)
        path_count = config.getint("general", "paths", fallback=0)
        for i in range(path_count):
            pos = flatten(
                config.getint("paths", f"{i}X"),
                config.getint("paths", f"{i}Y"),
                grid.width
            )
            obj = config.get("paths", f"{i}object")
            path = self.defaults_by_object[obj].copy()
            path.direction = config.getint("paths", f"{i}dir")
            grid.cells[pos].append(path)

        child_levels = {}

        # Add level objects & initialize level tree
        level_count = config.getint("general", "levels", fallback=0)
        for i in range(level_count):
            # Level colors can sometimes be omitted, defaults to white
            color = config.get("levels", f"{i}colour", fallback=None)
            if color is None:
                level = Item.level()
            else:
                c_0, c_1 = color.split(",")
                level = Item.level((int(c_0), int(c_1)))

            x = config.getint("levels", f"{i}X")  # no fallback
            y = config.getint("levels", f"{i}Y")  # if you can't locate it, it's fricked
            pos = flatten(x, y, grid.width)

            # # z mixed up with layer?
            # z = config.getint("levels", f"{i}Z", fallback=0)
            # level.layer = z

            grid.cells[pos].append(level)

            # level icons: the game handles them as special graphics
            # but the bot treats them as normal objects
            style = config.getint("levels", f"{i}style", fallback=0)
            number = config.getint("levels", f"{i}number", fallback=0)
            # "custom" style
            if style == -1:
                try:
                    icon = Item.icon(config.get("icons", f"{number}file"))
                    grid.cells[pos].append(icon)
                except configparser.NoSectionError:
                    # No icon exists for the level, I guess
                    pass
            # number style
            elif style == 0:
                if 0 <= number <= 99:
                    tens, ones = divmod(number, 10)
                    tens_icon = Item.icon(f"icon_default_tens_{tens}")
                    ones_icon = Item.icon(f"icon_default_ones_{ones}")
                    grid.cells[pos].append(tens_icon)
                    grid.cells[pos].append(ones_icon)
            # letter style
            elif style == 1:
                char = chr(number + ord('a'))
                if 'a' <= char <= 'z':
                    icon = Item.icon(f"icon_default_letter_{char}")
                    grid.cells[pos].append(icon)
            # "dot" style
            elif style == 2:
                if number >= 9:
                    icon = Item.icon("icon")
                else:
                    icon = Item.icon(f"icon_default_dot_{number + 1}")
                grid.cells[pos].append(icon)

            if initialize_level_tree and grid.map_id is not None:
                level_file = config.get("levels", f"{i}file")
                # Each level within
                child_levels[level_file] = (number, style)

        # Initialize the level tree
        # If map_id is None, then the levels are actually pointing back to this level's parent
        if initialize_level_tree and grid.map_id is not None:
            # specials are only used for special levels at the moment
            special_count = config.getint("general", "specials", fallback=0)
            for i in range(special_count):
                special_data = config.get("specials", f"{i}data")
                special_kind, *special_rest = special_data.split(",")
                if special_kind == "level":
                    # note: because of the comma separation these are still strings
                    level_file, style, number, *_ = special_rest
                    child = (int(number), int(style))
                    # print("adding spec to node", parent, grid.map_id, level_file, child)
                    child_levels[level_file] = child

            # merges both normal level & special level data together
            if child_levels:
                self.parent_levels[grid.filename] = (grid.map_id, child_levels)

        # Add background images
        image_count = config.getint("images", "total", fallback=0)
        for i in range(image_count):
            grid.images.append(config.get("images", str(i)))

        # Alternate would be to use changed_count & reading each record
        # The reason these aren't all just in `changed` is that MF2 limits
        # string sizes to 1000 chars or so.
        #
        # TODO: is it possible for `changed_short` to go over 1000 chars?
        # Probably not, since you'd need over 300 changed objects and I'm
        # not sure that's allowed by the editor (maybe file editing)
        #
        # `changed_short` exists for some custom levels
        changed_record = config.get("tiles", "changed_short", fallback=None)
        if changed_record is None:
            # levels in the base game (and custom levels without `changed_short`)
            # all provide `changed`, which CAN be an empty string
            # `split` doesn't filter out the empty string so this
            changed_record = config.get("tiles", "changed")
            changed_tiles = [x for x in changed_record.rstrip(",").split(",") if x != ""]
        else:
            changed_tiles = [f"object{x}" for x in changed_record.rstrip(",").split(",") if x != ""]

        # include only changes that will affect the visuals
        changes: dict[str, dict[str, Any]] = {tile: {} for tile in changed_tiles}
        attrs = ("image", "colour", "activecolour", "layer", "tiling")
        for tile in changed_tiles:
            for attr in attrs:
                # `tile` is of the form "objectXYZ", and
                new_attr = config.get("tiles", f"{tile}_{attr}", fallback=None)
                if new_attr is not None:
                    changes[tile][attr] = new_attr

        for cell in grid.cells:
            for item in cell:
                if item.obj in changes:
                    change = changes[item.obj]  # type: ignore
                    if "image" in change:
                        item.sprite = change["image"]
                    if "layer" in change:
                        item.layer = int(change["layer"])
                    if "tiling" in change:
                        item.tiling = int(change["tiling"])
                    # Text tiles always use their active color in renders,
                    # so `activecolour` is preferred over `colour`
                    #
                    # Including both active and inactive tiles would require
                    # the bot to parse the rules of the level, which is a
                    # lot of work for very little
                    #
                    # This unfortunately means that custom levels that use drastically
                    # different active & inactive colors will look different in renders
                    if "colour" in change:
                        x, y = change["colour"].split(",")
                        item.color = (int(x), int(y))
                    if "activecolour" in change and item.sprite is not None and item.sprite.startswith("text_"):
                        x, y = change["activecolour"].split(",")
                        item.color = (int(x), int(y))

        # Makes sure objects within a single cell are rendered in the right order
        # Items are sorted according to their layer attribute, in ascending order.
        for cell in grid.cells:
            cell.sort(key=lambda x: x.layer)

        return grid

    def read_layer(self, stream: BinaryIO, grid: Grid):
        buffer = stream.read(4)
        grid.width = int.from_bytes(buffer, byteorder="little")

        buffer = stream.read(4)
        grid.height = int.from_bytes(buffer, byteorder="little")

        size = grid.width * grid.height
        if size > 10000:
            raise ValueError(size)
        if grid.width > 1000:
            raise ValueError(size)
        if grid.height > 1000:
            raise ValueError(size)
        if len(grid.cells) == 0:
            for _ in range(size):
                grid.cells.append([])

        stream.read(32)  # don't care about these

        data_blocks = int.from_bytes(stream.read(1), byteorder="little")

        # MAIN
        stream.read(4)
        buffer = stream.read(4)
        compressed_size = int.from_bytes(buffer, byteorder="little")
        compressed = stream.read(compressed_size)

        zobj = zlib.decompressobj()
        map_buffer = zobj.decompress(compressed)

        items = []
        for j, k in enumerate(range(0, len(map_buffer), 2)):
            cell = grid.cells[j]
            id = int.from_bytes(map_buffer[k: k + 2], byteorder="little")

            item = self.defaults_by_id.get(id)
            if item is not None:
                item = item.copy()
            else:
                item = Item.empty()
                id = -1
            items.append(item)

            if id != -1:
                cell.append(item)

        if data_blocks == 2:
            # DATA
            stream.read(9)
            buffer = stream.read(4)
            compressed_size = int.from_bytes(buffer, byteorder="little") & (2 ** 32 - 1)

            zobj = zlib.decompressobj()
            dirs_buffer = zobj.decompress(stream.read(compressed_size))

            for j in range(len(dirs_buffer) - 1):
                try:
                    item = items[j]
                    item.direction = dirs_buffer[j]
                except IndexError:
                    # huh?
                    break
