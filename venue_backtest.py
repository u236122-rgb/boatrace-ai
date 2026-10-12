# -*- coding: utf-8 -*-
"""
venue_backtest.py ― 24場 × 春夏秋冬 の 正答率(1着的中率) と 回収率(上位3点) を出す
  ・2024年末まで学習 → 2025年以降を検証（全季節カバー・リーク無し）
  ・正答率＝モデルの1着予想が的中した割合
  ・回収率＝3連単「上位3点」を1点100円で買った場合の払戻/投資（確定配当payouts）
  本番モデルは変更しない（検証専用）。展示なし＝通常設定。
"""
import sys, io
import numpy as np
import pandas as pd
import lightgbm as lgb
import build_model as bm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def log(m): print(m, flush=True)

SPLIT = "2025-01-01"   # これ以降を検証
MONTHS = list(range(1, 13))
VN = bm.VENUE_NAMES
FEATS = [f for f in bm.FEATURES if "exhibit" not in f]   # 展示は使わない（通常設定）
CATEG = [c for c in bm.CATEG if c in FEATS]

log("データ準備 ...")
entries = pd.read_csv("data/csv/entries.csv")
results = pd.read_csv("data/csv/results.csv")
payouts = pd.read_csv("data/csv/payouts.csv")
res = results[["boat", "date", "venue", "race", "course", "st", "finish",
               "weather", "wind_dir", "wind", "wave"]].copy()
df = entries.merge(res, on=["boat", "date", "venue", "race"], how="inner")
df = df.dropna(subset=["finish"]).reset_index(drop=True)
df["target"] = (df["finish"] == 1).astype(int)
df["win"] = df["target"]; df["top2"] = (df["finish"] <= 2).astype(int)
df["st_num"] = pd.to_numeric(df["st"], errors="coerce")
df = bm.prep_common(df)
df = df.sort_values("date").reset_index(drop=True)

src_cols = ["racer_id", "course", "venue", "date", "win", "top2", "st_num"]
tr_mask = df["date"] < SPLIT
src = df[tr_mask][src_cols]
glob = bm.build_glob(df[tr_mask][src_cols])
log(f"学習 {tr_mask.sum():,}行（〜2024末） / 検証 {(~tr_mask).sum():,}行（2025〜）")

log("特徴量付与（LOO/素引き）...")
tr = bm.attach_features(df[tr_mask], src, glob, loo=True)
va = bm.attach_features(df[~tr_mask], src, glob, loo=False)

log("学習中 ...")
k = int(len(tr) * 0.97)
dtr = lgb.Dataset(tr.iloc[:k][FEATS], tr.iloc[:k]["target"], categorical_feature=CATEG,
                  free_raw_data=False)
des = lgb.Dataset(tr.iloc[k:][FEATS], tr.iloc[k:]["target"], reference=dtr,
                  categorical_feature=CATEG, free_raw_data=False)
params = dict(bm.PARAMS); params["verbose"] = -1
model = lgb.train(params, dtr, 3000, valid_sets=[des], valid_names=["es"],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
va = va.copy()
va["pred"] = model.predict(va[FEATS], num_iteration=model.best_iteration)

# 払戻（確定3連単）
pay = payouts.set_index(["date", "venue", "race"])
pay_combo = pay["combo"].to_dict()
pay_yen = pay["payout"].to_dict()


def trifecta_top3(strength):
    s = np.clip(strength, 1e-9, None); S = s.sum()
    out = []
    for a in range(6):
        for b in range(6):
            if b == a: continue
            sab = S - s[a]
            for c in range(6):
                if c == a or c == b: continue
                pr = (s[a]/S) * (s[b]/sab) * (s[c]/(sab - s[b]))
                out.append((pr, f"{a+1}-{b+1}-{c+1}"))
    out.sort(reverse=True)
    return [c for _, c in out[:3]]


log("検証レースを集計中 ...")
rec = {}   # (venue, month) -> [n, top1_hit, t3_cost, t3_ret, t3_hit]
for (dt, v, rc), g in va.groupby(["date", "venue", "race"]):
    g = g.sort_values("boat")
    if len(g) != 6:
        continue
    key3 = (dt, v, rc)
    mon = int(dt[5:7])
    st = g["pred"].values
    strength = st / st.sum()
    top1 = int(strength.argmax()) + 1
    winner = int(g[g["target"] == 1]["boat"].iloc[0]) if (g["target"] == 1).any() else None
    cell = rec.setdefault((v, mon), [0, 0, 0, 0, 0])
    cell[0] += 1
    if winner == top1:
        cell[1] += 1
    if key3 in pay_combo:
        buy = trifecta_top3(strength)
        cell[2] += 300
        if pay_combo[key3] in buy:
            cell[3] += int(pay_yen[key3])
            cell[4] += 1


MIN_N = 100   # 信頼性のため、その月この場でこのレース数以上を対象
log(f"\n===== 各月 信頼度(正答率)が高い場 トップ5（各月{MIN_N}R以上の場のみ）=====")
log("  正答率を優先で並べ、回収率(上位3点)も併記。回収率100%未満=損。")
for m in MONTHS:
    rows = []
    for v in range(1, 25):
        c = rec.get((v, m))
        if not c or c[0] < MIN_N:
            continue
        acc = c[1] / c[0] * 100
        roi = c[3] / c[2] * 100 if c[2] else None
        rows.append((acc, roi, v, c[0]))
    rows.sort(key=lambda x: -x[0])
    log(f"\n【{m}月】")
    for rank, (acc, roi, v, n) in enumerate(rows[:5], 1):
        rs = f"{roi:4.0f}%" if roi is not None else "  -  "
        log(f"  {rank}. {VN[v]:<5} 正答率 {acc:4.1f}%  回収率 {rs}  (n={n})")

N = sum(c[0] for c in rec.values()); H = sum(c[1] for c in rec.values())
CO = sum(c[2] for c in rec.values()); RE = sum(c[3] for c in rec.values())
log(f"\n全体：検証{N:,}レース／正答率{H/N*100:.1f}%／上位3点回収率{RE/CO*100:.1f}%")
log("※2025年以降を検証。回収率は3連単で振れ大、月別は誤差大（参考）。")
