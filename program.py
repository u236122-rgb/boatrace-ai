# -*- coding: utf-8 -*-
"""
program.py  ―  当日の番組表(出走表)を読み込むパーサ
------------------------------------------------------------
公式の番組表ファイル data/raw/B/bYYMMDD.txt（Shift-JIS）を解析し、
その日の全場・全レースの「6艇の選手登録番号」を取り出す。
GUIの「当日まとめ予想」から使う。
"""
import os
import re
import glob

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_B = os.path.join(BASE, "data", "raw", "B")

_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")
_RE_BGN = re.compile(r"^(\d\d)BBGN")
_RE_END = re.compile(r"^(\d\d)BEND")
_RE_RACE = re.compile(r"([０-９]{1,2})Ｒ")
_RE_RACER = re.compile(r"^([1-6])\s+(\d{4})")
# モーター番号まで取る詳細版（取れれば節間モーター成績に使う）
_RE_RACER_FULL = re.compile(
    r"^([1-6])\s+(\d{4})\D+?\d{2}\D+?\d{2}(?:A1|A2|B1|B2)\s+"
    r"[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(\d+)\s+\d+\.\d{2}\s*\d+\s+\d+\.\d{2}")


def list_program_dates():
    """利用可能な番組表の日付(YYMMDD)を新しい順で返す。"""
    files = glob.glob(os.path.join(RAW_B, "b??????.txt"))
    dates = sorted((os.path.basename(f)[1:7] for f in files), reverse=True)
    return dates


def yymmdd_to_iso(yymmdd):
    return f"20{yymmdd[0:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"


def parse_program(yymmdd):
    """指定日(YYMMDD)の番組表を解析し、レースのリストを返す。
    各レース = {venue, race, title, racers:[{boat,racer_id}*6]}"""
    path = os.path.join(RAW_B, f"b{yymmdd}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="cp932", errors="replace") as f:
        text = f.read()

    races = []
    venue = None
    cur = None
    for line in text.splitlines():
        m = _RE_BGN.match(line)
        if m:
            venue = int(m.group(1))
            cur = None
            continue
        if _RE_END.match(line):
            venue = None
            cur = None
            continue
        if venue is None:
            continue
        # レース見出し（全角数字+Ｒ、かつ距離/締切を含む行）
        rm = _RE_RACE.search(line)
        if rm and ("ｍ" in line or "締切" in line):
            race_no = int(rm.group(1).translate(_ZEN))
            cur = {"venue": venue, "race": race_no,
                   "title": line.strip(), "racers": []}
            races.append(cur)
            continue
        # 選手行（まずモーター番号付きで試し、ダメなら登番のみ）
        rf = _RE_RACER_FULL.match(line)
        if rf and cur is not None and len(cur["racers"]) < 6:
            cur["racers"].append({"boat": int(rf.group(1)),
                                  "racer_id": int(rf.group(2)),
                                  "motor_no": int(rf.group(3))})
            continue
        rr = _RE_RACER.match(line)
        if rr and cur is not None and len(cur["racers"]) < 6:
            cur["racers"].append({"boat": int(rr.group(1)),
                                  "racer_id": int(rr.group(2)),
                                  "motor_no": None})

    # 6艇そろったレースだけ採用
    races = [r for r in races if len(r["racers"]) == 6]
    races.sort(key=lambda r: (r["venue"], r["race"]))
    return races


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    dates = list_program_dates()
    print("利用可能な日付(新しい順):", dates[:5], "...計", len(dates))
    d = dates[0]
    races = parse_program(d)
    print(f"\n{yymmdd_to_iso(d)} の番組: {len(races)}レース")
    print("場一覧:", sorted({r['venue'] for r in races}))
    r = races[0]
    print("例:", "場", r["venue"], r["race"], "R",
          [x["racer_id"] for x in r["racers"]])
    print("   ", r["title"])
