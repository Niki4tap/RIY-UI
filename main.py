#!/usr/bin/python
# -*- coding: utf-8 -*-
import ui
import asyncio

import argparse
from os.path import isdir, exists
from io import BytesIO
from render.cogs.global_cog import GlobalCog
from render.cogs.reload import ReloadCog
from render.db import Database
from time import time
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QColor, QMovie
from PyQt5.QtCore import QByteArray, QBuffer, QIODevice
from sys import exit as sys_exit, argv as sys_argv
from traceback import format_tb
from PIL import Image, ImageSequence


def dump_exception(e, ui_i):
    tb = exception_to_traceback(e)
    for s in tb:
        print(s)

    ui_i.textOutput.setTextColor(QColor("#FF0000"))
    for s in tb:
        ui_i.textOutput.append(s)
    ui_i.textOutput.setTextColor(QColor("#FFFFFF"))


def exception_to_traceback(e):
    ret = [type(e).__name__ + ": " + str(e)]
    ret.extend(format_tb(e.__traceback__))
    return ret


class Renderer:
    def __init__(self):
        self.event_loop = asyncio.new_event_loop()
        self.block_on = self.event_loop.run_until_complete
        self.db = Database()
        self.reloading = False
        self.reload_cog = ReloadCog(self.db)
        self.global_cog = GlobalCog(self.db)
        self.block_on(self.db.connect("robot.db"))


class GIFDisplay:
    def __init__(self):
        self.qbytes = QByteArray()
        self.qbuffer = QBuffer()
        self.buffer = BytesIO()
        self.qmovie = QMovie()

    def assign_buffer(self, buf):
        self.qmovie.stop()
        self.buffer = buf
        self.qbytes = QByteArray(self.buffer.getvalue())
        self.qbuffer = QBuffer(self.qbytes)
        self.qbuffer.open(QIODevice.ReadOnly)

    def display(self, label):
        self.qmovie.setFormat(QByteArray(b"GIF"))
        self.qmovie.setDevice(self.qbuffer)
        label.setMovie(self.qmovie)
        self.qmovie.start()


class GUIApp:
    def __init__(self):
        self.renderer = Renderer()
        self.app = QApplication(sys_argv)
        self.window = QMainWindow()
        self.ui_i = ui.Ui_MainWindow()
        self.gifdisp = GIFDisplay()

    def reload(self):
        if self.renderer.reloading:
            self.ui_i.textOutput.setTextColor(QColor("#FF0000"))
            self.ui_i.textOutput.append("Already reloading!")
            self.ui_i.textOutput.setTextColor(QColor("#FFFFFF"))
            return

        self.ui_i.textOutput.setTextColor(QColor("#FFAA00"))
        self.ui_i.textOutput.append("Reloading...")
        self.ui_i.textOutput.setTextColor(QColor("#FFFFFF"))

        self.renderer.reloading = True
        start_time = time()
        self.renderer.block_on(self.renderer.reload_cog.loaddata())
        self.renderer.block_on(self.renderer.reload_cog.loadletters())
        delta = time() - start_time
        self.renderer.reloading = False

        self.ui_i.textOutput.append(f"Done in {delta:.2f} seconds.")

    def exec(self):
        self.window.setFixedSize(800, 650)
        self.ui_i.setupUi(self.window)

        self.ui_i.modeSelector.insertItem(0, "Auto")
        self.ui_i.modeSelector.insertItem(1, "Rule")
        self.ui_i.modeSelector.insertItem(2, "Tile")
        self.ui_i.modeSelector.insertItem(3, "Level")
        self.ui_i.modeSelector.insertItem(4, "Mobile")

        self.ui_i.runButton.clicked.connect(self.rendering)
        self.ui_i.reloadButton.clicked.connect(self.reload)

        self.window.show()
        self.reload()
        sys_exit(self.app.exec())

    def render(self):
        try:
            full_command = self.ui_i.commandInput.toPlainText()
            if self.ui_i.modeSelector.currentIndex() == 0:
                full_command = full_command[1:]
                if full_command[:4] == "rule":
                    buffer = self.renderer.block_on(self.renderer.global_cog.render_tiles(objects=full_command[4:], is_rule=True))
                elif full_command[:4] == "tile":
                    buffer = self.renderer.block_on(self.renderer.global_cog.render_tiles(objects=full_command[4:], is_rule=False))
                elif full_command[:5] == "level":
                    buffer, info = self.renderer.block_on(self.renderer.global_cog.level_command(query=full_command[5:]))
                elif full_command[:6] == "mobile":
                    buffer, info = self.renderer.block_on(self.renderer.global_cog.mobile(query=full_command[6:]))
                else:
                    raise RuntimeError("Auto mode was not able to recognize the command")

            elif self.ui_i.modeSelector.currentIndex() == 1:
                buffer = self.renderer.block_on(self.renderer.global_cog.render_tiles(objects=full_command, is_rule=True))
            elif self.ui_i.modeSelector.currentIndex() == 2:
                buffer = self.renderer.block_on(self.renderer.global_cog.render_tiles(objects=full_command, is_rule=False))
            elif self.ui_i.modeSelector.currentIndex() == 3:
                buffer, info = self.renderer.block_on(self.renderer.global_cog.level_command(query=full_command))
            elif self.ui_i.modeSelector.currentIndex() == 4:
                buffer, info = self.renderer.block_on(self.renderer.global_cog.mobile(query=full_command))
            else:
                raise RuntimeError("Command not found/implemented")
        except Exception as e:
            dump_exception(e, self.ui_i)
            return None, None
        return buffer, info if "info" in locals() else None

    def rendering(self):
        if self.renderer.reloading:
            self.ui_i.textOutput.setTextColor(QColor("#FF0000"))
            self.ui_i.textOutput.append("Can't run render! Reloading in progress.")
            self.ui_i.textOutput.setTextColor(QColor("#FFFFFF"))
            return

        self.ui_i.textOutput.append("Running...")

        start_time = time()
        buffer, info = self.render()
        delta = time() - start_time

        if not isinstance(buffer, str):
            buffer.seek(0)
            im = Image.open(buffer)
            resized_frames = []
            if self.ui_i.resizeCheck.isChecked():
                if im.size[0] > 770 and not im.size[1] > 370:
                    for frame in ImageSequence.Iterator(im):
                        resized_frames.append(frame.resize((770, im.size[1]), Image.ANTIALIAS))
                elif not im.size[0] > 770 and im.size[1] > 370:
                    for frame in ImageSequence.Iterator(im):
                        resized_frames.append(frame.resize((im.size[0], 370), Image.ANTIALIAS))
                elif im.size[0] > 770 and im.size[1] > 370:
                    for frame in ImageSequence.Iterator(im):
                        resized_frames.append(frame.resize((770, 370), Image.ANTIALIAS))
                else:
                    resized_frames = [frame for frame in ImageSequence.Iterator(im)]
            else:
                resized_frames = [frame for frame in ImageSequence.Iterator(im)]
            try:
                buf = BytesIO()
                resized_frames[0].save(buf, format="GIF", save_all=True, append_images=[frame for frame in resized_frames[1:]])
                self.gifdisp.assign_buffer(buf)
                self.gifdisp.display(self.ui_i.imageView)
            except Exception as e:
                dump_exception(e, self.ui_i)
                return
            open("output.gif", "wb").write(buffer.read())
            if "info" in locals():
                self.ui_i.textOutput.setTextColor(QColor("#AAAAFF"))
                self.ui_i.textOutput.append(info)
                self.ui_i.textOutput.setTextColor(QColor("#FFFFFF"))
            self.ui_i.textOutput.append("Saved to output.gif")
            self.ui_i.textOutput.append(f"Done in {delta:.2f} seconds.")
        else:
            self.ui_i.textOutput.setTextColor(QColor("#FF0000"))
            self.ui_i.textOutput.append(buffer)
            self.ui_i.textOutput.setTextColor(QColor("#FFFFFF"))


class CLIApp:
    def __init__(self):
        if not(isdir("data") and exists("data")):
            print("Failed to find \"data\" directory, you need to either get it from the game or download from the bot's source: https://github.com/RocketRace/robot-is-you/tree/master/data")
            sys_exit(1)

        parser = argparse.ArgumentParser(description='Robot is you UI. For GUI, launch without arguments')
        parser.add_argument('-v', '--version', action='version', version="RIY UI 0.1")
        parser.add_argument('-c', '--command')
        parser.add_argument('-t', '--type')
        self.args = vars(parser.parse_args())

    def init_renderer(self):
        self.renderer = Renderer()

    def render(self):
        try:
            full_command = self.args["command"]
            if self.args["type"] == "auto":
                full_command = full_command[1:]
                if full_command[:4] == "rule":
                    full_command = full_command.replace("LF", "\n")
                    buffer = self.renderer.block_on(self.renderer.global_cog.render_tiles(objects=full_command[4:], is_rule=True))
                elif full_command[:4] == "tile":
                    full_command = full_command.replace("LF", "\n")
                    buffer = self.renderer.block_on(self.renderer.global_cog.render_tiles(objects=full_command[4:], is_rule=False))
                elif full_command[:5] == "level":
                    buffer, info = self.renderer.block_on(self.renderer.global_cog.level_command(query=full_command[5:]))
                elif full_command[:6] == "mobile":
                    buffer, info = self.renderer.block_on(self.renderer.global_cog.mobile(query=full_command[6:]))
                else:
                    raise RuntimeError("Auto mode was not able to recognize the command")

            elif self.args["type"] == "rule":
                full_command = full_command.replace("LF", "\n")
                buffer = self.renderer.block_on(self.renderer.global_cog.render_tiles(objects=full_command, is_rule=True))
            elif self.args["type"] == "tile":
                full_command = full_command.replace("LF", "\n")
                buffer = self.renderer.block_on(self.renderer.global_cog.render_tiles(objects=full_command, is_rule=False))
            elif self.args["type"] == "level":
                buffer, info = self.renderer.block_on(self.renderer.global_cog.level_command(query=full_command))
            elif self.args["type"] == "mobile":
                buffer, info = self.renderer.block_on(self.renderer.global_cog.mobile(query=full_command))
            else:
                raise RuntimeError("Command not found/implemented")
        except Exception as e:
            tb = exception_to_traceback(e)
            for s in tb:
                print(s)
            return None, None
        return buffer, info if "info" in locals() else None

    def main(self):
        if self.args == {"command": None, "type": None}:
            app = GUIApp()
            app.exec()
        else:
            if self.args["command"] is not None and self.args["type"] is None:
                print("Command was supplied, but type wasn't\n"
                      "-t, --type: \"auto\", \"rule\", \"tile\", \"level\", \"mobile\"\n"
                      "Assuming \"auto\"")
                self.args["type"] = "auto"
            elif self.args["command"] is None and self.args["type"] is not None:
                print("Type was supplied, but command wasn't\n"
                      "-c, --command: \"+rule baba\"")
                sys_exit(1)
            self.init_renderer()
            start_time = time()
            buffer, info = self.render()
            delta = time() - start_time
            if not isinstance(buffer, str):
                if info is not None:
                    print(info)
                open("output.gif", "wb").write(buffer.read())
                print("Saved to output.gif\n"
                      f"Done in {delta:.2f} seconds.")
            else:
                print(buffer)


def main():
    cli = CLIApp()
    cli.main()


if __name__ == "__main__":
    main()
