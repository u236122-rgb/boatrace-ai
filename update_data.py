# -*- coding: utf-8 -*-
"""
update_data.py  ―  競艇データ更新ツール
============================================================
公式サイトから生データ(番組表B・競走成績K)をダウンロード＆展開し、
data/csv/entries.csv・results.csv・payouts.csv を最新まで追記する。

  python update_data.py            … 不足日を自動取得してCSV更新
  python update_data.py validate   … 既存CSVと突き合わせて正しさ確認
  python update_data.py 260615 260621  … 指定範囲だけ取得・更新

生データURL（公式）:
  番組表 B: http://www1.mbrace.or.jp/od2/B/YYYYMM/bYYMMDD.lzh
  成績   K: http://www1.mbrace.or.jp/od2/K/YYYYMM/kYYMMDD.lzh
"""
import os
import re
import sys
import io
import csv
import glob
import datetime
import urllib.request

import lhafile

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_B = os.path.join(BASE, "data", "raw", "B")
RAW_K = os.path.join(BASE, "data", "raw", "K")
CSV = os.path.join(BASE, "data", "csv")
URL_B = "http://www1.mbrace.or.jp/od2/B/{ym}/b{d}.lzh"
URL_K = "http://www1.mbrace.or.jp/od2/K/{ym}/k{d}.lzh"

WEATHER = ["晴", "曇り", "雨", "雪", "霧"]
ENT_COLS = ["boat", "racer_id", "name", "age", "branch", "weight", "grade",
            "win_nat", "top2_nat", "win_loc", "top2_loc", "motor_no",
            "motor_top2", "boat_no", "boat_top2", "date", "venue", "race"]
RES_COLS = ["boat", "racer_id", "name", "motor", "boatno", "exhibit", "course",
            "st", "racetime", "date", "venue", "race", "finish", "abnormal",
            "weather", "wind_dir", "wind", "wave"]
PAY_COLS = ["date", "venue", "race", "combo", "payout"]

_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")
RE_BBGN = re.compile(r"^(\d\d)BBGN")
RE_KBGN = re.compile(r"^(\d\d)KBGN")
RE_END = re.compile(r"^(\d\d)(BEND|KEND)")
RE_BRACE = re.compile(r"([０-９]{1,2})Ｒ")
# 出走表(B)の選手行
RE_B = re.compile(
    r"^([1-6])\s+(\d{4})(\D+?)(\d{2})(\D+?)(\d{2})(A1|A2|B1|B2)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"(\d+)\s+(\d+\.\d{2})\s*(\d+)\s+(\d+\.\d{2})")
# 成績(K)の着順行  着 艇 登番 名前 ﾓｰﾀｰ ﾎﾞｰﾄ 展示 進入 ST ﾚｰｽﾀｲﾑ
RE_K = re.compile(
    r"^\s*(\d\d|[A-Z]\d?)\s+([1-6])\s+(\d{4})\s+(.+?)\s+(\d{1,3})\s+(\d{1,3})\s+"
    r"(\d+\.\d{2})\s+([1-6])\s+([0-9A-Z.]+)\s+(.*)$")
# 欠場など、展示・進入が無い不完全行（K1 など）
RE_K_DNS = re.compile(
    r"^\s*([0-9A-Z]\d?)\s+([1-6])\s+(\d{4})\s+(.+?)\s+(\d{1,3})\s+(\d{1,3})\b")
# K のレース見出し（条件つき）
RE_KHEAD = re.compile(
    r"(\d{1,2})R\s+.*?H\d+m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm")
# K の払戻金行（3連単）
RE_PAY = re.compile(r"^\s*(\d{1,2})R\s+(\d-\d-\d)\s+(\d+)")


def yymmdd_iso(d):
    return f"20{d[0:2]}-{d[2:4]}-{d[4:6]}"


def clean_name(s):
    return s.replace("　", "").replace(" ", "")


# ---------------- 出走表(B) → entries 行 ----------------
def parse_b(path):
    with open(path, encoding="cp932", errors="replace") as f:
        text = f.read()
    base = os.path.basename(path)
    date = yymmdd_iso(base[1:7])
    rows = []
    venue = race = None
    for line in text.splitlines():
        m = RE_BBGN.match(line)
        if m:
            venue = int(m.group(1)); race = None; continue
        if RE_END.match(line):
            venue = None; continue
        if venue is None:
            continue
        rm = RE_BRACE.search(line)
        if rm and ("ｍ" in line or "締切" in line):
            race = int(rm.group(1).translate(_ZEN)); continue
        b = RE_B.match(line)
        if b and race is not None:
            g = b.groups()
            rows.append({
                "boat": int(g[0]), "racer_id": int(g[1]),
                "name": clean_name(g[2]), "age": int(g[3]),
                "branch": g[4].strip(), "weight": int(g[5]), "grade": g[6],
                "win_nat": float(g[7]), "top2_nat": float(g[8]),
                "win_loc": float(g[9]), "top2_loc": float(g[10]),
                "motor_no": int(g[11]), "motor_top2": float(g[12]),
                "boat_no": int(g[13]), "boat_top2": float(g[14]),
                "date": date, "venue": venue, "race": race,
            })
    return rows


# ---------------- 成績(K) → results 行 ＆ payouts 行 ----------------
def parse_k(path):
    with open(path, encoding="cp932", errors="replace") as f:
        text = f.read()
    base = os.path.basename(path)
    date = yymmdd_iso(base[1:7])
    results, payouts = [], []
    venue = None
    cond = {}   # race -> (weather,wind_dir,wind,wave)
    cur_race = None
    for line in text.splitlines():
        m = RE_KBGN.match(line)
        if m:
            venue = int(m.group(1)); cur_race = None; cond = {}; continue
        if RE_END.match(line):
            venue = None; continue
        if venue is None:
            continue
        # 払戻金（3連単）
        pm = RE_PAY.match(line)
        if pm and "-" in pm.group(2):
            payouts.append({"date": date, "venue": venue,
                            "race": int(pm.group(1)), "combo": pm.group(2),
                            "payout": int(pm.group(3))})
            continue
        # レース見出し（条件）
        norm = line.replace("　", " ")
        hm = RE_KHEAD.search(norm)
        if hm:
            cur_race = int(hm.group(1))
            cond[cur_race] = (hm.group(2), hm.group(3),
                              int(hm.group(4)), int(hm.group(5)))
            continue
        if cur_race is None:
            continue
        w = cond.get(cur_race, ("", "", "", ""))
        # 着順行（通常）
        km = RE_K.match(line)
        if km:
            g = km.groups()
            chaku = g[0]
            finish = int(chaku) if chaku.isdigit() else ""
            abnormal = "" if chaku.isdigit() else chaku
            results.append({
                "boat": int(g[1]), "racer_id": int(g[2]),
                "name": clean_name(g[3]), "motor": int(g[4]),
                "boatno": int(g[5]), "exhibit": float(g[6]),
                "course": int(g[7]), "st": g[8], "racetime": g[9].strip(),
                "date": date, "venue": venue, "race": cur_race,
                "finish": finish, "abnormal": abnormal,
                "weather": w[0], "wind_dir": w[1], "wind": w[2], "wave": w[3],
            })
            continue
        # 欠場など不完全行（展示・進入なし）
        dm = RE_K_DNS.match(line)
        if dm and not dm.group(1).isdigit():
            g = dm.groups()
            results.append({
                "boat": int(g[1]), "racer_id": int(g[2]),
                "name": clean_name(g[3]), "motor": int(g[4]),
                "boatno": int(g[5]), "exhibit": "", "course": "",
                "st": "", "racetime": "", "date": date, "venue": venue,
                "race": cur_race, "finish": "", "abnormal": g[0],
                "weather": w[0], "wind_dir": w[1], "wind": w[2], "wave": w[3],
            })
    return results, payouts


# ---------------- 検証（既存CSVと突き合わせ）----------------
def validate():
    import pandas as pd
    print("=== 検証：既存CSVと自作パーサの突き合わせ ===")
    e = pd.read_csv(os.path.join(CSV, "entries.csv"))
    rb = parse_b(os.path.join(RAW_B, "b260614.txt"))
    eb = pd.DataFrame(rb)
    eold = e[e["date"] == "2026-06-14"].reset_index(drop=True)
    print(f"entries 6/14  既存{len(eold)}行 / 自作{len(eb)}行")
    key = ["venue", "race", "boat"]
    merged = eold.merge(eb, on=key, suffixes=("_old", "_new"))
    diff = 0
    for col in ["racer_id", "name", "age", "weight", "grade", "win_nat",
                "top2_nat", "win_loc", "top2_loc", "motor_no", "motor_top2",
                "boat_no", "boat_top2"]:
        a = merged[f"{col}_old"].astype(str).str.replace(r"\.0$", "", regex=True)
        b = merged[f"{col}_new"].astype(str).str.replace(r"\.0$", "", regex=True)
        d = (a != b).sum()
        if d:
            diff += d
            print(f"  ⚠ {col}: {d}件 不一致 例:",
                  list(zip(a[a != b].head(2), b[a != b].head(2))))
    print("entries 一致" if diff == 0 else f"entries 不一致 計{diff}")

    r = pd.read_csv(os.path.join(CSV, "results.csv"))
    rk, pk = parse_k(os.path.join(RAW_K, "k260613.txt"))
    rkd = pd.DataFrame(rk)
    rold = r[r["date"] == "2026-06-13"].reset_index(drop=True)
    print(f"\nresults 6/13  既存{len(rold)}行 / 自作{len(rkd)}行")
    m2 = rold.merge(rkd, on=["venue", "race", "boat"], suffixes=("_old", "_new"))
    diff2 = 0
    for col in ["racer_id", "motor", "boatno", "exhibit", "course",
                "finish", "abnormal", "weather", "wind_dir", "wind", "wave"]:
        a = m2[f"{col}_old"].astype(str).str.replace(".0", "", regex=False)
        b = m2[f"{col}_new"].astype(str).str.replace(".0", "", regex=False)
        a = a.replace("nan", ""); b = b.replace("nan", "")
        d = (a != b).sum()
        if d:
            diff2 += d
            print(f"  ⚠ {col}: {d}件 不一致 例:",
                  list(zip(a[a != b].head(3), b[a != b].head(3))))
    print("results 一致" if diff2 == 0 else f"results 不一致 計{diff2}")
    pay = pd.read_csv(os.path.join(CSV, "payouts.csv"))
    payold = pay[pay["date"] == "2026-06-13"]
    print(f"\npayouts 6/13  既存{len(payold)}行 / 自作{len(pk)}行")


# ---------------- ダウンロード＆展開 ----------------
def _is_placeholder(txt):
    """確定前にDLされた仮ファイル（結果が空）かどうか。"""
    try:
        with open(txt, encoding="cp932", errors="replace") as f:
            s = f.read()
        return ("終了後に登録" in s) or ("登録されます" in s)
    except Exception:
        return False


def ensure_raw(kind, d, force=False):
    """kind='B'/'K', d='YYMMDD'。生txtが無ければ公式から落として展開。
    既存でも仮ファイル(未確定)や force 指定なら取り直す。
    取得できたら txtパス、無ければ None。"""
    raw_dir = RAW_B if kind == "B" else RAW_K
    txt = os.path.join(raw_dir, f"{kind.lower()}{d}.txt")
    if os.path.exists(txt) and not force and not _is_placeholder(txt):
        return txt
    ym = "20" + d[0:2] + d[2:4]
    url = (URL_B if kind == "B" else URL_K).format(ym=ym, d=d)
    try:
        data = urllib.request.urlopen(url, timeout=30).read()
    except Exception:
        return None
    lzh = txt + ".lzh"
    os.makedirs(raw_dir, exist_ok=True)
    with open(lzh, "wb") as f:
        f.write(data)
    arc = None
    try:
        arc = lhafile.Lhafile(lzh)
        content = arc.read(arc.namelist()[0])
        with open(txt, "wb") as f:
            f.write(content)
    except Exception:
        txt = None
    finally:
        try:
            if arc is not None:
                arc.fp.close()
        except Exception:
            pass
        try:
            os.remove(lzh)
        except Exception:
            pass
    return txt


def _yd(yymmdd):
    return datetime.date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))


def _drange(start_iso, end_d):
    cur = _yd(start_iso[2:].replace("-", "")) + datetime.timedelta(days=1)
    end = _yd(end_d)
    while cur <= end:
        yield cur.strftime("%y%m%d")
        cur += datetime.timedelta(days=1)


def latest_raw_date():
    ds = []
    for pat in (os.path.join(RAW_B, "b??????.txt"), os.path.join(RAW_K, "k??????.txt")):
        for f in glob.glob(pat):
            ds.append(os.path.basename(f)[1:7])
    return max(ds) if ds else datetime.date.today().strftime("%y%m%d")


def update():
    import pandas as pd
    print("=== データ更新開始 ===")
    ent = pd.read_csv(os.path.join(CSV, "entries.csv"))
    res = pd.read_csv(os.path.join(CSV, "results.csv"))
    pay = pd.read_csv(os.path.join(CSV, "payouts.csv"))
    e_dates, r_dates = set(ent["date"]), set(res["date"])
    base = min(ent["date"].max(), res["date"].max())   # ここ以降を点検
    end_d = max(datetime.date.today().strftime("%y%m%d"), latest_raw_date())
    print(f"entries 最新 {ent['date'].max()} / results 最新 {res['date'].max()} "
          f"/ 取得上限 {yymmdd_iso(end_d)}")

    new_e, new_r, new_p = [], [], []
    for d in _drange(base, end_d):
        iso = yymmdd_iso(d)
        if iso not in e_dates:                 # 出走表：未取得の日だけ
            t = ensure_raw("B", d)
            if t:
                rows = parse_b(t)
                if rows:
                    new_e += rows
                    print(f"  [B] {iso} … {len(rows)}行")
        if iso not in r_dates:                 # 結果：未取得の日だけ（仮ファイルは取り直す）
            t = ensure_raw("K", d)
            if t:
                rr, pp = parse_k(t)
                if rr:
                    new_r += rr
                    new_p += pp
                    print(f"  [K] {iso} … 結果{len(rr)}行 / 払戻{len(pp)}行")

    if not new_e and not new_r:
        print("追加データはありません（すでに最新）。")
        return

    def append(df_old, rows, cols, key, path):
        if not rows:
            return df_old, 0
        add = pd.DataFrame(rows)[cols]
        merged = pd.concat([df_old, add], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
        n = len(merged) - len(df_old)
        merged.to_csv(path, index=False, encoding="utf-8")
        return merged, n

    import shutil
    for path in ("entries.csv", "results.csv", "payouts.csv"):
        p = os.path.join(CSV, path)
        if not os.path.exists(p + ".orig"):
            shutil.copy(p, p + ".orig")   # 最初の1回だけ原本を保存

    _, ne = append(ent, new_e, ENT_COLS, ["date", "venue", "race", "boat"],
                   os.path.join(CSV, "entries.csv"))
    _, nr = append(res, new_r, RES_COLS, ["date", "venue", "race", "boat"],
                   os.path.join(CSV, "results.csv"))
    _, npay = append(pay, new_p, PAY_COLS, ["date", "venue", "race", "combo"],
                     os.path.join(CSV, "payouts.csv"))
    print(f"\n追加: entries +{ne} / results +{nr} / payouts +{npay} 行")
    print("完了！ このあと 1_モデル学習.bat を実行すると最新データで学習されます。")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate()
    else:
        update()
