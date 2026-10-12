# -*- coding: utf-8 -*-
"""
exp_ml_pitfalls.py ― GeminiのML落とし穴のうち、未測定のものをシミュレーション
  ④ コンセプトドリフト：古いデータだけで学習したモデルは、最近(2025)で劣化するか？
       old(2021-2022)学習 vs recent(2023-2024)学習 vs full(2021-2024)学習 を
       すべて 2025以降 で検証して比較。
  ① 未学習(アンダーフィット)：わざと単純すぎるモデルにすると精度が落ちることを確認
       （＝今のモデルは単純すぎ＝未学習ではない、の裏取り）。
  全て 2025以降で検証（リーク無し）。本番モデルは不変。
"""
import sys, io
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
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
MK = list(zip(df["date"].astype(str), df["venue"].astype(int),
              df["race"].astype(int), df["boat"].astype(int)))
MT = dict(zip(MK, df["motor_setsu_top2"])); MN = dict(zip(MK, df["motor_setsu_n"]))


def attach_motor(f):
    key = list(zip(f["date"].astype(str), f["venue"].astype(int),
                   f["race"].astype(int), f["boat"].astype(int)))
    f["motor_setsu_top2"] = [MT[k] for k in key]
    f["motor_setsu_n"] = [MN[k] for k in key]


test_mask = df["date"] >= "2025-01-01"


def run(train_mask, params, label, rounds=3000, es=True):
    glob_ = bm.build_glob(df[train_mask][sc])
    tr = bm.attach_features(df[train_mask], df[train_mask][sc], glob_, loo=True)
    va = bm.attach_features(df[test_mask], df[train_mask][sc], glob_, loo=False)
    attach_motor(tr); attach_motor(va)
    if es:
        k = int(len(tr) * 0.97)
        d = lgb.Dataset(tr.iloc[:k][FE], tr.iloc[:k]["target"], categorical_feature=CA, free_raw_data=False)
        e = lgb.Dataset(tr.iloc[k:][FE], tr.iloc[k:]["target"], reference=d, categorical_feature=CA, free_raw_data=False)
        m = lgb.train(params, d, rounds, valid_sets=[e], valid_names=["es"],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        bi = m.best_iteration
    else:
        d = lgb.Dataset(tr[FE], tr["target"], categorical_feature=CA, free_raw_data=False)
        m = lgb.train(params, d, rounds)
        bi = rounds
    va = va.copy(); va["pred"] = m.predict(va[FE], num_iteration=bi)
    auc = roc_auc_score(va["target"], va["pred"])
    n = h = 0
    for _, g in va.groupby(["date", "venue", "race"]):
        g = g.sort_values("boat")
        if len(g) != 6 or not (g["target"] == 1).any():
            continue
        n += 1
        h += int(int(g["pred"].values.argmax()) + 1 ==
                 int(g[g["target"] == 1]["boat"].iloc[0]))
    log(f"  [{label}] 2025的中率={h/n*100:.1f}%  AUC={auc:.4f}  (n={n:,})")
    return h/n*100, auc


P = dict(bm.PARAMS); P["verbose"] = -1
TINY = dict(objective="binary", metric="auc", learning_rate=0.1,
            num_leaves=2, min_data_in_leaf=20000, verbose=-1)

log("\n===== ④ コンセプトドリフト（学習期間を変えて2025を予想）=====")
old = run((df["date"] >= "2021-01-01") & (df["date"] < "2023-01-01"), P, "古い 2021-2022 学習")
rec = run((df["date"] >= "2023-01-01") & (df["date"] < "2025-01-01"), P, "最近 2023-2024 学習")
ful = run(df["date"] < "2025-01-01", P, "全部 2021-2024 学習(本番)")

log("\n===== ① 未学習（わざと単純すぎるモデル）=====")
und = run(df["date"] < "2025-01-01", TINY, "極小モデル(葉2・40本)", rounds=40, es=False)

log("\n================= まとめ =================")
log(f"  ④ドリフト: 古い学習 {old[0]:.1f}% → 最近学習 {rec[0]:.1f}% → 全部 {ful[0]:.1f}%")
log(f"     （古いデータだけだと {ful[0]-old[0]:+.1f}pt の劣化＝ドリフトの大きさ）")
log(f"  ①未学習 : 極小モデル {und[0]:.1f}% vs 本番 {ful[0]:.1f}%（{und[0]-ful[0]:+.1f}pt）")
log("     本番が極小より大きく上＝『単純すぎ(未学習)ではない』の裏取り。")
log("\n※リーク無し（各モデルとも自分の学習期間の実績のみ使用→2025検証）。本番不変。")
