# -*- coding: utf-8 -*-
"""
exp_overfit_sim.py ― 「過学習を考慮しない場合」のシミュレーション
  ① 正直版：2024末まで学習→2025以降で検証＋LOO（自分除外）＝今の本番のやり方
  ② 幻版  ：全データで学習し“同じデータ”を予想＋LOOなし（自分の結果も特徴に混入）
            ＝過学習・リークを放置した場合に画面に出る“嘘の好成績”
  的中率と単勝回収率を両方で出し、差＝過学習の幻 を可視化する。
"""
import sys, io
import numpy as np
import pandas as pd
import lightgbm as lgb
import build_model as bm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def log(m): print(m, flush=True)

log("データ準備 ...")
entries = pd.read_csv("data/csv/entries.csv")
results = pd.read_csv("data/csv/results.csv")
res = results[["boat", "date", "venue", "race", "course", "st", "finish",
               "weather", "wind_dir", "wind", "wave"]].copy()
df = entries.merge(res, on=["boat", "date", "venue", "race"], how="inner")
df = df.dropna(subset=["finish"]).reset_index(drop=True)
df["target"] = (df["finish"] == 1).astype(int)
df["win"] = df["target"]; df["top2"] = (df["finish"] <= 2).astype(int)
df["st_num"] = pd.to_numeric(df["st"], errors="coerce")
df = bm.prep_common(df)
df = df.sort_values(["date", "venue", "race", "boat"]).reset_index(drop=True)
df = bm.add_motor_setsu(df)
FE = [f for f in bm.FEATURES if "exhibit" not in f]
CA = [c for c in bm.CATEG if c in FE]
sc = ["racer_id", "course", "venue", "date", "win", "top2", "st_num"]

# 単勝の実払戻
pall = pd.read_csv("data/csv/payouts_all.csv")
tan = pall[pall["bet_type"] == "単勝"]
TAN = {(d, int(v), int(r)): (c, int(p)) for d, v, r, c, p in
       zip(tan["date"], tan["venue"], tan["race"], tan["combo"], tan["payout"])}


def attach_motor(frame):
    key = list(zip(frame["date"].astype(str), frame["venue"].astype(int),
                   frame["race"].astype(int), frame["boat"].astype(int)))
    frame["motor_setsu_top2"] = [MT[k] for k in key]
    frame["motor_setsu_n"] = [MN[k] for k in key]
MK = list(zip(df["date"].astype(str), df["venue"].astype(int),
              df["race"].astype(int), df["boat"].astype(int)))
MT = dict(zip(MK, df["motor_setsu_top2"])); MN = dict(zip(MK, df["motor_setsu_n"]))


def evaluate(va):
    """va に pred 済み。的中率と単勝回収率（全体／自信0.7+）を返す。"""
    n = h = 0
    cost = ret = 0
    chc = chh = ccost = cret = 0
    for (dt, v, r), g in va.groupby(["date", "venue", "race"]):
        g = g.sort_values("boat")
        if len(g) != 6 or not (g["target"] == 1).any():
            continue
        raw = np.clip(g["pred"].values, 1e-9, None)
        s = raw / raw.sum()
        top = int(np.argmax(s))
        topboat = top + 1
        conf = float(s.max())
        winner = int(g[g["target"] == 1]["boat"].iloc[0])
        n += 1; h += int(topboat == winner)
        w = TAN.get((dt, int(v), int(r)))
        if w:
            cost += 100; ret += (w[1] if w[0] == str(topboat) else 0)
            if conf >= 0.7:
                ccost += 100; cret += (w[1] if w[0] == str(topboat) else 0)
                chc += 1; chh += int(topboat == winner)
    return (h/n*100, ret/cost*100 if cost else 0,
            cret/ccost*100 if ccost else 0, n, chc)


pa = dict(bm.PARAMS); pa["verbose"] = -1

log("\n① 正直版（時系列分割＋LOO）を学習・検証 ...")
trm = df["date"] < "2025-01-01"
glob_h = bm.build_glob(df[trm][sc])
tr_h = bm.attach_features(df[trm], df[trm][sc], glob_h, loo=True)
va_h = bm.attach_features(df[~trm], df[trm][sc], glob_h, loo=False)
attach_motor(tr_h); attach_motor(va_h)
k = int(len(tr_h)*0.97)
d1 = lgb.Dataset(tr_h.iloc[:k][FE], tr_h.iloc[:k]["target"], categorical_feature=CA, free_raw_data=False)
e1 = lgb.Dataset(tr_h.iloc[k:][FE], tr_h.iloc[k:]["target"], reference=d1, categorical_feature=CA, free_raw_data=False)
m1 = lgb.train(pa, d1, 3000, valid_sets=[e1], valid_names=["es"], callbacks=[lgb.early_stopping(80, verbose=False)])
va_h = va_h.copy(); va_h["pred"] = m1.predict(va_h[FE], num_iteration=m1.best_iteration)
hon = evaluate(va_h)

log("② 幻版（全データ学習＋LOOなし＋同一データを予想）を学習・検証 ...")
glob_a = bm.build_glob(df[sc])
allf = bm.attach_features(df, df[sc], glob_a, loo=False)   # ← LOOなし＝自分の結果が混入
attach_motor(allf)
d2 = lgb.Dataset(allf[FE], allf["target"], categorical_feature=CA, free_raw_data=False)
m2 = lgb.train(pa, d2, 1500)                                # 早期終了なし＝木を伸ばす
allf = allf.copy(); allf["pred"] = m2.predict(allf[FE])     # ← 学習に使った同じデータを予想
fan = evaluate(allf)

log("\n================= 結果：過学習の『幻』 =================")
log(f"{'':<26}{'的中率':>9}{'単勝回収':>10}{'単勝回収(自信0.7+)':>18}")
log(f"{'① 正直版(本番のやり方)':<24}{hon[0]:>8.1f}%{hon[1]:>9.1f}%{hon[2]:>16.1f}%")
log(f"{'② 幻版(過学習を放置)':<25}{fan[0]:>8.1f}%{fan[1]:>9.1f}%{fan[2]:>16.1f}%")
log(f"\n  差（幻のゲタ）：的中 +{fan[0]-hon[0]:.1f}pt / 単勝回収 +{fan[1]-hon[1]:.1f}pt "
    f"/ 自信0.7+回収 +{fan[2]-hon[2]:.1f}pt")
log("\n※②は『当ててから予想している』状態。画面では絶好調に見えるが、本番(未来)では①に戻る。")
log("  この差こそ、時系列分割＋LOOで潰している“過学習の幻”。")
