# -*- coding: utf-8 -*-
"""
odds_collector.py  ―  3連単オッズ自動収集デーモン
============================================================
boatrace.jp から各レースの3連単120通りのオッズを、
締切の【15分前・13分前・10分前・5分前】の4回取得して保存する。

  ・締切時刻は番組表(B)の「電話投票締切予定HH:MM」から取得
  ・保存先 data/odds/  ファイル名: YYYYMMDD_場_RR_NNmin.csv
  ・各行に「締切何分前か(mins_before)」「取得時刻(fetched_at)」も記録
  ・取得成否は data/odds/_collector.log に記録
  ・サーバ負荷軽減のためアクセス間隔を空ける
  ・既に取得済み(ファイルあり)はスキップ＝再起動しても重複しない

PCを起動しっぱなしにし、タスクスケジューラで常駐させると放置で溜まる。
手動テスト:  python odds_collector.py            （常駐）
            python odds_collector.py once       （今この瞬間に取れる分だけ1回）
"""
import os
import re
import sys
import csv
import time
import random
import logging
import datetime
import urllib.request

import program
import update_data

BASE = os.path.dirname(os.path.abspath(__file__))
ODDS_DIR = os.path.join(BASE, "data", "odds")
os.makedirs(ODDS_DIR, exist_ok=True)
LOG_PATH = os.path.join(ODDS_DIR, "_collector.log")

MINS_BEFORE = [15, 13, 10, 5]      # 締切の何分前に取るか
MIN_REQUEST_GAP = 3.0              # HTTPアクセスの最低間隔(秒)＝サーバ負荷軽減
FIRE_GRACE = 100                   # 予定時刻からこの秒数までなら取得実行
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
ODDS_URL = "https://www.boatrace.jp/owpc/pc/race/odds3t?rno={r}&jcd={v:02d}&hd={hd}"
ZEN = str.maketrans("０１２３４５６７８９", "0123456789")

logging.basicConfig(
    filename=LOG_PATH, level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logging.getLogger().addHandler(_console)
log = logging.getLogger().info


# 3連単120通りの「ページ上の並び順」（列=1着, 各列 2着昇順×3着昇順、行優先）
def _combo_order():
    cols = []
    for ic in range(1, 7):
        cc = [(ic, ni, sa)
              for ni in range(1, 7) if ni != ic
              for sa in range(1, 7) if sa != ic and sa != ni]
        cols.append(cc)
    return [cols[c][r] for r in range(20) for c in range(6)]


COMBOS = _combo_order()
_last_req = [0.0]


def fetch_odds(date_iso, venue, race):
    """3連単120通りの {(1,2,3): odds} を返す。取れなければ例外。"""
    gap = MIN_REQUEST_GAP - (time.time() - _last_req[0])
    if gap > 0:
        time.sleep(gap)
    hd = date_iso.replace("-", "")
    url = ODDS_URL.format(r=race, v=venue, hd=hd)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "replace")
    _last_req[0] = time.time()
    cells = re.findall(r'<td class="oddsPoint[^"]*"[^>]*>(.*?)</td>', html, re.S)
    if len(cells) != 120:
        raise ValueError(f"oddsセル数={len(cells)}(期待120)")
    out = {}
    for combo, raw in zip(COMBOS, cells):
        t = re.sub(r"<[^>]*>", "", raw).strip()
        try:
            out[combo] = float(t)
        except ValueError:
            out[combo] = ""      # 欠場・未発売など
    return out


def save_odds(date_iso, venue, race, mins, odds_map):
    fetched = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hd = date_iso.replace("-", "")
    path = os.path.join(ODDS_DIR, f"{hd}_{venue:02d}_{race}R_{mins:02d}min.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "venue", "race", "mins_before", "fetched_at",
                    "combo", "odds"])
        for combo, od in odds_map.items():
            w.writerow([date_iso, venue, race, mins, fetched,
                        "%d-%d-%d" % combo, od])
    return path


def parse_deadline(title, day):
    """番組表の見出しから締切時刻(datetime)を取り出す。"""
    norm = title.translate(ZEN)
    m = re.search(r"締切予定\s*(\d{1,2})\s*[：:]\s*(\d{2})", norm)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    return datetime.datetime(day.year, day.month, day.day, hh, mm)


def load_plan(day):
    """その日の (取得予定時刻, 場, R, 分前) のリストを作る。"""
    yymmdd = day.strftime("%y%m%d")
    txt = update_data.ensure_raw("B", yymmdd)
    if not txt:
        log(f"番組表が取得できません: {yymmdd}")
        return []
    plan = []
    for r in program.parse_program(yymmdd):
        dl = parse_deadline(r.get("title", ""), day)
        if dl is None:
            continue
        for m in MINS_BEFORE:
            plan.append({"time": dl - datetime.timedelta(minutes=m),
                         "venue": r["venue"], "race": r["race"], "mins": m,
                         "date": day.isoformat(), "deadline": dl})
    plan.sort(key=lambda p: p["time"])
    return plan


def out_path(p):
    hd = p["date"].replace("-", "")
    return os.path.join(ODDS_DIR, f'{hd}_{p["venue"]:02d}_{p["race"]}R_{p["mins"]:02d}min.csv')


def do_fetch(p):
    try:
        odds = fetch_odds(p["date"], p["venue"], p["race"])
        n_ok = sum(1 for v in odds.values() if v != "")
        path = save_odds(p["date"], p["venue"], p["race"], p["mins"], odds)
        log(f"OK 場{p['venue']:02d} {p['race']}R 締切{p['mins']}分前 "
            f"有効{n_ok}/120 → {os.path.basename(path)}")
        return True
    except Exception as e:
        log(f"NG 場{p['venue']:02d} {p['race']}R 締切{p['mins']}分前 失敗: {e}")
        return False


def run_once():
    """今この瞬間、取得ウィンドウ内のものを取得（テスト用）。"""
    day = datetime.date.today()
    now = datetime.datetime.now()
    plan = load_plan(day)
    fired = 0
    for p in plan:
        if os.path.exists(out_path(p)):
            continue
        if -FIRE_GRACE <= (now - p["time"]).total_seconds() <= FIRE_GRACE:
            do_fetch(p)
            fired += 1
    log(f"run_once 完了: {fired}件取得")


def main():
    log("=== オッズ収集デーモン開始 ===")
    cur_day = None
    plan = []
    while True:
        try:
            today = datetime.date.today()
            if today != cur_day:
                cur_day = today
                plan = load_plan(today)
                log(f"{today} の予定 {len(plan)}件を読み込み")

            now = datetime.datetime.now()
            pending = [p for p in plan
                       if not os.path.exists(out_path(p))
                       and (now - p["time"]).total_seconds() <= FIRE_GRACE]
            if not pending:
                time.sleep(60)        # 当日分なし→1分後に再評価（日付変更も拾う）
                continue

            nxt = min(pending, key=lambda p: p["time"])
            wait = (nxt["time"] - datetime.datetime.now()).total_seconds()
            if wait > 0:
                time.sleep(min(wait + random.uniform(0, 3), 120))  # 細切れに待つ
                continue

            # 取得ウィンドウ内 → 実行（1回失敗したら20秒後に1度だけ再試行）
            if not do_fetch(nxt):
                time.sleep(20)
                do_fetch(nxt)
        except Exception as e:           # 予期せぬ例外でも止めない
            log(f"ループ例外（継続します）: {e}")
            time.sleep(30)
        # do_fetch でファイルが出来れば pending から自然に外れる。
        # 失敗で未生成のままなら次ループで FIRE_GRACE を過ぎて除外される。


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        run_once()
    else:
        main()
