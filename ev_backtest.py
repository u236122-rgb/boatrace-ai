# -*- coding: utf-8 -*-
"""
ev_backtest.py ― 真のEV検証（モデル確率 × 実・直前オッズ）
============================================================
オッズ収集(data/odds)で貯めた『全120通りの実オッズ』と、確定結果を突き合わせ、
  EV = モデル確率 × 実オッズ（締切直前スナップショット）
で EV>閾値 の目だけ買った場合の回収率を検証する。

Geminiの罠への対応：
  罠1: 締切“直前(5分前)”のオッズで判定（確定配当ではなく実際に買える値）
  罠3: 本命/中穴/大穴のオッズ帯ごとに回収率をクロス集計
  罠2: 高配当帯はオッズインパクトの注意を併記（市場規模が小さい）

使い方：
  ・オッズ収集を回し続け、レース後に 5_データ更新.bat で結果を取り込む
  ・データが貯まったら  python ev_backtest.py
  本番モデルは変更しない（検証専用）。

【重要な注意：3連単確率の較正】
  1着(単勝)の確率は較正済み(予測≒実績)だが、3連単の“同時確率”は
  Plackett-Luceの独立仮定で歪み、中穴の確率が過大評価されやすい。
  実際、6/23の実オッズで試算するとEV>1が33/120点・EV最大5.9倍と非現実的に多発。
  → EV>1 をそのまま買うと『見かけのEV』を大量に拾う恐れ。
  本検証では、EVだけでなく『オッズ帯別の実回収率』で“本当に勝てる帯”を確認し、
  必要なら3連単確率の較正(温度スケーリング/Isotonic)を後段で追加する。
"""
import os
import sys
import io
import glob
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ODDS = os.path.join(BASE, "data", "odds")
SNAPSHOT_PRIORITY = ["05", "10", "13", "15"]   # 直前(5分前)を最優先
EV_THRESHOLDS = [1.0, 1.1, 1.3, 1.5]


def log(m): print(m, flush=True)


def band(odds):
    if odds < 20: return "本命(〜20倍)"
    if odds < 50: return "中穴(20-50倍)"
    if odds < 100: return "大穴(50-100倍)"
    return "超大穴(100倍〜)"


def load_odds_races():
    """data/odds から (date,venue,race)->{combo:odds} を、締切直前スナップショットで作る。"""
    files = glob.glob(os.path.join(ODDS, "*min.csv"))
    chosen = {}   # (hd,venue,race) -> mins(優先度) , path
    for f in files:
        base = os.path.basename(f)[:-4]            # 20260623_02_1R_05min
        try:
            hd, v, rno, mn = base.split("_")
            mins = mn.replace("min", "")
            key = (hd, int(v), int(rno.replace("R", "")))
        except Exception:
            continue
        pr = SNAPSHOT_PRIORITY.index(mins) if mins in SNAPSHOT_PRIORITY else 99
        if key not in chosen or pr < chosen[key][0]:
            chosen[key] = (pr, f)
    races = {}
    for key, (_, f) in chosen.items():
        d = pd.read_csv(f)
        od = {}
        for _, r in d.iterrows():
            try:
                od[str(r["combo"])] = float(r["odds"])
            except (ValueError, TypeError):
                pass
        if od:
            races[key] = od
    return races


def main():
    if not os.path.isdir(ODDS):
        log("data/odds がありません。先にオッズ収集を動かしてください。")
        return
    races = load_odds_races()
    log(f"オッズ取得済みレース: {len(races):,}（締切直前スナップショット採用）")
    if not races:
        log("まだオッズデータがありません。収集が貯まってから再実行してください。")
        return

    # 結果（確定3連単）と照合用
    pay = pd.read_csv(os.path.join(BASE, "data", "csv", "payouts.csv"))
    # オッズ名は YYYYMMDD(8桁) なので照合キーも8桁に揃える
    pay["key"] = list(zip(pay["date"].str.replace("-", ""), pay["venue"], pay["race"]))
    win_combo = dict(zip(pay["key"], pay["combo"]))

    # 出走メンバー・条件は結果(results.csv)から取得
    from core import Predictor
    pred = Predictor()
    results = pd.read_csv(os.path.join(BASE, "data", "csv", "results.csv"),
                          usecols=["date", "venue", "race", "boat", "racer_id",
                                   "weather", "wind_dir", "wind", "wave", "finish"])

    def trifecta_probs(strength):
        s = np.clip(strength, 1e-9, None); S = s.sum()
        out = {}
        for a in range(6):
            for b in range(6):
                if b == a: continue
                sab = S - s[a]
                for c in range(6):
                    if c == a or c == b: continue
                    out[f"{a+1}-{b+1}-{c+1}"] = (s[a]/S)*(s[b]/sab)*(s[c]/(sab-s[b]))
        return out

    matched, pending = [], 0
    for (hd, v, rno), od in races.items():
        iso = f"{hd[:4]}-{hd[4:6]}-{hd[6:8]}"   # 20260623 -> 2026-06-23
        wc = win_combo.get((hd, v, rno))
        sub = results[(results["date"] == iso) & (results["venue"] == v) & (results["race"] == rno)]
        if wc is None or len(sub) != 6:        # 結果未取り込み → 保留
            pending += 1
            continue
        sub = sub.sort_values("boat")
        ids = sub["racer_id"].astype(int).tolist()
        w = sub.iloc[0]
        r = pred.predict(v, ids, str(w["weather"]), str(w["wind_dir"]),
                         float(w["wind"]) if pd.notna(w["wind"]) else 0,
                         float(w["wave"]) if pd.notna(w["wave"]) else 0)
        probs = trifecta_probs(r["strength"])
        for combo, p in probs.items():
            if combo in od:
                matched.append({"combo": combo, "p": p, "odds": od[combo],
                                "ev": p * od[combo], "won": combo == wc})

    log(f"結果と照合できたレース由来の買い目候補: {len(matched):,}点 / 結果待ち {pending} レース")
    if not matched:
        log("\nまだ『オッズ＋確定結果』が揃ったレースがありません。")
        log("→ レース後に 5_データ更新.bat で結果を取り込み、後日再実行してください。")
        log("（オッズ収集の仕組み・モデル予想・EV計算の枠組みは動作確認済み）")
        return

    M = pd.DataFrame(matched)
    M["band"] = M["odds"].apply(band)
    log("\n===== 真のEV検証：EV>閾値 の目だけ買った回収率 =====")
    log(f"{'EV閾値':>6}{'買い点数':>8}{'的中':>7}{'回収率':>8}")
    for t in EV_THRESHOLDS:
        sel = M[M["ev"] >= t]
        if len(sel) == 0:
            continue
        cost = len(sel) * 100
        ret = (sel["won"] * sel["odds"] * 100).sum()
        hit = sel["won"].sum()
        log(f"{t:>6}{len(sel):>8}{hit/len(sel)*100:6.1f}%{ret/cost*100:7.1f}%")

    log("\n===== オッズ帯ごとの回収率（EV>1.0で購入・罠3対応）=====")
    sel = M[M["ev"] >= 1.0]
    for b in ["本命(〜20倍)", "中穴(20-50倍)", "大穴(50-100倍)", "超大穴(100倍〜)"]:
        s2 = sel[sel["band"] == b]
        if len(s2) == 0:
            continue
        cost = len(s2) * 100
        ret = (s2["won"] * s2["odds"] * 100).sum()
        log(f"  {b:<14} 買い{len(s2):>5}点 的中{s2['won'].sum()/len(s2)*100:5.1f}% "
            f"回収率{ret/cost*100:6.1f}%")
    log("\n※ 罠1:直前5分前オッズで判定済み。罠2:大穴帯は自分の投票でオッズが下がる")
    log("  リスクあり（市場小）。回収率100%超でも、点数が少なければ誤差として扱うこと。")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
