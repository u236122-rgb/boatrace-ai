# -*- coding: utf-8 -*-
"""
app.py  ―  競艇AI予想ツールの起動用（英語ファイル名・バッチ起動を確実にする）
日本語ファイル名(予想GUI.py)はバッチから渡すと文字化けする場合があるため、
ASCII名のこのファイル経由で GUI を起動する。中身は 予想GUI.py を呼ぶだけ。
"""
import os
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

spec = importlib.util.spec_from_file_location(
    "yosou_gui", os.path.join(HERE, "予想GUI.py"))
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)

if __name__ == "__main__":
    gui.App().mainloop()
