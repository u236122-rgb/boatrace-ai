# -*- coding: utf-8 -*-
"""
exp_simplify.py ― 「単純・強正則のモデルの方が頑健か」を複数年×複数seedで検証
  発見：極小モデルが2025で本番をわずかに上回った（1年・1seed）。本当か？
  ・ホールドアウト2年：2024（2021-23学習）／2025（2021-24学習）
  ・設定4段階：現行(複雑)→中→小→極小(stump)
  ・各seed[1,7,42]で平均。特徴量は年ごとに1回だけ作って使い回す。
  本番モデルは不変。リーク無し（各年とも過去のみ学習→当年検証）。
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


def base(**kw):
    p = dict(objective="binary", metric="auc",
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
    p.update(kw); return p


CONFIGS = [
    ("現行(複雑)", base(learning_rate=0.03, num_leaves=127, min_data_in_leaf=2000, lambda_l2=10.0), 3000, True),
    ("中",        base(learning_rate=0.03, num_leaves=31,  min_data_in_leaf=5000, lambda_l2=20.0), 3000, True),
    ("小(強正則)", base(learning_rate=0.05, num_leaves=15,  min_data_in_leaf=10000, lambda_l2=30.0), 3000, True),
    ("極小(stump)", base(learning_rate=0.1, num_leaves=2,   min_data_in_leaf=20000), 60, False),
]
SEEDS = [1, 7, 42]
HOLDOUTS = [("2024", ("2021-01-01", "2024-01-01"), ("2024-01-01", "2025-01-01")),
            ("2025", ("2021-01-01", "2025-01-01"), ("2025-01-01", "2099-01-01"))]


def eval_race(va, pred):
    va = va.assign(pred=pred)
    n = h = 0
    for _, g in va.groupby(["date", "venue", "race"]):
        g = g.sort_values("boat")
        if len(g) != 6 or not (g["target"] == 1).any():
            continue
        n += 1
        h += int(int(g["pred"].values.argmax()) + 1 ==
                 int(g[g["target"] == 1]["boat"].iloc[0]))
    return h / n * 100


results_acc = {c[0]: {} for c in CONFIGS}
results_auc = {c[0]: {} for c in CONFIGS}

for yr, (t0, t1), (e0, e1) in HOLDOUTS:
    log(f"\n=== ホールドアウト {yr}（特徴量を構築中…）===")
    trm = (df["date"] >= t0) & (df["date"] < t1)
    tem = (df["date"] >= e0) & (df["date"] < e1)
    glob_ = bm.build_glob(df[trm][sc])
    tr = bm.attach_features(df[trm], df[trm][sc], glob_, loo=True); attach_motor(tr)
    va = bm.attach_features(df[tem], df[trm][sc], glob_, loo=False); attach_motor(va)
    k = int(len(tr) * 0.97)
    Xtr, ytr = tr.iloc[:k][FE], tr.iloc[:k]["target"]
    Xes, yes = tr.iloc[k:][FE], tr.iloc[k:]["target"]
    Xva, yva = va[FE], va["target"]
    for name, params, rounds, es in CONFIGS:
        accs, aucs = [], []
        for sd in SEEDS:
            p = dict(params); p["seed"] = sd
            if es:
                d = lgb.Dataset(Xtr, ytr, categorical_feature=CA, free_raw_data=False)
                e = lgb.Dataset(Xes, yes, reference=d, categorical_feature=CA, free_raw_data=False)
                m = lgb.train(p, d, rounds, valid_sets=[e], valid_names=["es"],
                              callbacks=[lgb.early_stopping(80, verbose=False)])
                bi = m.best_iteration
            else:
                d = lgb.Dataset(Xtr, ytr, categorical_feature=CA, free_raw_data=False)
                m = lgb.train(p, d, rounds); bi = rounds
            pr = m.predict(Xva, num_iteration=bi)
            aucs.append(roc_auc_score(yva, pr))
            accs.append(eval_race(va, pr))
        results_acc[name][yr] = float(np.mean(accs))
        results_auc[name][yr] = float(np.mean(aucs))
        log(f"  [{name:<11}] {yr} 的中率 {np.mean(accs):.2f}%  AUC {np.mean(aucs):.4f}  "
            f"(seed幅 ±{np.std(accs):.2f}pt)")

log("\n================= まとめ（seed平均）=================")
log(f"{'設定':<13}{'2024的中':>9}{'2025的中':>9}{'平均的中':>9}{'平均AUC':>10}")
for name, *_ in CONFIGS:
    a24, a25 = results_acc[name]["2024"], results_acc[name]["2025"]
    u = (results_auc[name]["2024"] + results_auc[name]["2025"]) / 2
    log(f"{name:<12}{a24:>8.2f}%{a25:>8.2f}%{(a24+a25)/2:>8.2f}%{u:>10.4f}")

cur = results_acc["現行(複雑)"]
log("\n--- 現行との差（プラス＝単純な方が良い）---")
for name, *_ in CONFIGS[1:]:
    d24 = results_acc[name]["2024"] - cur["2024"]
    d25 = results_acc[name]["2025"] - cur["2025"]
    both = "★両年で勝ち" if d24 > 0 and d25 > 0 else ""
    log(f"  {name:<11}: 2024 {d24:+.2f}pt / 2025 {d25:+.2f}pt  {both}")
log("\n※両年＋seed平均で一貫して現行を上回る設定があれば、単純化は本物。")
log("※リーク無し・本番モデル不変。")
