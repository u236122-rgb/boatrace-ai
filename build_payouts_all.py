# -*- coding: utf-8 -*-
"""
build_payouts_all.py ― 既存のK票から「全券種の払戻＋人気」と「決まり手」を取り出す。
  既に持っているデータ/raw/K/k*.txt を読み直すだけ（再ダウンロード不要）。
  既存の payouts.csv（3連単のみ）は壊さず、別ファイルを新規作成する:
    ・payouts_all.csv … date,venue,race,bet_type,combo,payout,pop（人気）
        bet_type = 単勝 / 複勝 / 2連単 / 2連複 / 拡連複 / 3連単 / 3連複
    ・kimarite.csv    … date,venue,race,kimarite（逃げ/差し/まくり 等）
実行:  python build_payouts_all.py
"""
import os
import re
import csv
import sys
import io
import glob
import update_data as ud

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
RAW_K = os.path.join(BASE, "data", "raw", "K")
CSV = os.path.join(BASE, "data", "csv")

# 券種ラベル（全角）→ 出力名
LABELS = [("単勝", "単勝"), ("複勝", "複勝"), ("２連単", "2連単"),
          ("２連複", "2連複"), ("拡連複", "拡連複"),
          ("３連単", "3連単"), ("３連複", "3連複")]
# 組番 払戻 [人気 N] の繰り返し（単勝/複勝は人気なし、複勝は1行2件）
RE_PAIR = re.compile(r"(\d(?:-\d){0,2})\s+(\d+)(?:\s+人気\s+(\d+))?")


def parse_payouts(path):
    text = open(path, encoding="cp932", errors="replace").read()
    date = ud.yymmdd_iso(os.path.basename(path)[1:7])
    venue = cur_race = cur_bet = None
    pays, kima = [], []
    for line in text.splitlines():
        if ud.RE_KBGN.match(line):
            venue = int(ud.RE_KBGN.match(line).group(1))
            cur_race = cur_bet = None
            continue
        if ud.RE_END.match(line):
            venue = cur_race = cur_bet = None
            continue
        if venue is None:
            continue
        norm = line.replace("　", " ")
        hm = ud.RE_KHEAD.search(norm)
        if hm:                                   # レース見出し
            cur_race = int(hm.group(1)); cur_bet = None
            continue
        # 決まり手（「着 艇 登番 … ﾚｰｽﾀｲﾑ ◯◯」の最後の語）
        if cur_race is not None and "ﾚｰｽﾀｲﾑ" in line:
            toks = norm.split()
            km = toks[-1] if toks and toks[-1] != "ﾚｰｽﾀｲﾑ" else ""
            kima.append((date, venue, cur_race, km))
            continue
        stripped = norm.strip()
        # ラベル付き払戻行
        lab = fw = None
        for f, nm in LABELS:
            if stripped.startswith(f):
                lab, fw = nm, f; break
        if lab and cur_race is not None:
            cur_bet = lab
            for combo, pay, pop in RE_PAIR.findall(stripped[len(fw):]):
                pays.append((date, venue, cur_race, lab, combo,
                             int(pay), int(pop) if pop else ""))
            continue
        # 拡連複は2件目・3件目がラベルなしで続く
        if cur_bet == "拡連複" and cur_race is not None and re.match(r"^\d-\d\s+\d", stripped):
            for combo, pay, pop in RE_PAIR.findall(stripped):
                pays.append((date, venue, cur_race, "拡連複", combo,
                             int(pay), int(pop) if pop else ""))
            continue
    return pays, kima


def main():
    files = sorted(glob.glob(os.path.join(RAW_K, "k??????.txt")))
    print(f"K票 {len(files):,} 日分を再パースします（再DLなし）...")
    all_p, all_k = [], []
    for i, f in enumerate(files):
        try:
            p, k = parse_payouts(f)
            all_p.extend(p); all_k.extend(k)
        except Exception as e:
            print(f"  スキップ {os.path.basename(f)}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} 日 ... 払戻{len(all_p):,}行")

    pa = os.path.join(CSV, "payouts_all.csv")
    with open(pa, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["date", "venue", "race", "bet_type", "combo", "payout", "pop"])
        w.writerows(all_p)
    ka = os.path.join(CSV, "kimarite.csv")
    with open(ka, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["date", "venue", "race", "kimarite"])
        w.writerows(all_k)

    print(f"\n✅ 完了")
    print(f"  payouts_all.csv : {len(all_p):,} 行")
    print(f"  kimarite.csv    : {len(all_k):,} 行")
    # 券種別の件数
    from collections import Counter
    c = Counter(r[3] for r in all_p)
    print("  券種別:", "  ".join(f"{k}={v:,}" for k, v in c.items()))
    ck = Counter(r[3] for r in all_k if r[3])
    print("  決まり手:", "  ".join(f"{k}={v:,}" for k, v in ck.most_common()))


if __name__ == "__main__":
    main()
