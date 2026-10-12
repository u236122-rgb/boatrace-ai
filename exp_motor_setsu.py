# -*- coding: utf-8 -*-
"""
exp_motor_setsu.py ― 「節間モーター成績」を特徴量に足すと精度が上がるか検証
  アイデア：同じ節（連続開催）の中で、そのモーターがここまで何連対しているか
            （＝仕上がり）を“前日まで”の結果だけで集計し、特徴量に追加。
  比較：ベースライン  vs  ベースライン＋節間モーター成績
        2024末まで学習→2025以降で検証（リーク防止：当日結果は使わない）。
  本番モデル(build_model.py)は一切いじらない。効けば後で採用を検討する。
"""
import sys, io, os
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

# ---- 節（連続開催）IDを場ごとに付与（3日以上空いたら別の節）----
log("節（開催単位）を判定 ...")
ser = {}
for v, gv in df.groupby("venue"):
    dts = sorted(gv["date"].unique())
    sid, prev = 0, None
    for d in dts:
        if prev is not None and (pd.to_datetime(d) - pd.to_datetime(prev)).days > 3:
            sid += 1
        ser[(v, d)] = sid
        prev = d
df["series"] = [ser[(v, d)] for v, d in zip(df["venue"], df["date"])]

# ---- 節間モーター成績（前日までの2連率）を集計（リーク防止）----
log("節間モーター2連率（前日まで）を集計 ...")
key = ["venue", "series", "motor_no"]
daily = (df.groupby(key + ["date"])
           .agg(d_sum=("top2", "sum"), d_cnt=("top2", "size"))
           .reset_index().sort_values(key + ["date"]))
g = daily.groupby(key)
daily["prior_sum"] = g["d_sum"].cumsum() - daily["d_sum"]   # 当日を除く
daily["prior_cnt"] = g["d_cnt"].cumsum() - daily["d_cnt"]
daily["m_setsu"] = daily["prior_sum"] / daily["prior_cnt"]
df = df.merge(daily[key + ["date", "prior_cnt", "m_setsu"]],
              on=key + ["date"], how="left")

df["motor_setsu_n"] = df["prior_cnt"].fillna(0)
# 前日までの実績が無い（節初日）→ 番組表のモーター2連率(%→比率)で代用
df["motor_setsu_top2"] = df["m_setsu"]
mask0 = df["motor_setsu_n"] == 0
df.loc[mask0, "motor_setsu_top2"] = df.loc[mask0, "motor_top2"] / 100.0
df["motor_setsu_top2"] = df["motor_setsu_top2"].fillna(df["motor_top2"].mean() / 100.0)

log(f"  節初日(実績なし)の割合 = {mask0.mean()*100:.1f}%  "
    f"平均節間レース数 = {df['motor_setsu_n'].mean():.1f}")

# ---- 特徴量を構築（LOO）----
log("特徴量を構築（学習はLOOでリーク防止）...")
trm = df["date"] < "2025-01-01"
sc = ["racer_id", "course", "venue", "date", "win", "top2", "st_num"]
src = df[trm][sc]
glob_ = bm.build_glob(df[trm][sc])
FE = [f for f in bm.FEATURES if "exhibit" not in f]      # 展示は今回不使用
CA = [c for c in bm.CATEG if c in FE]
tr = bm.attach_features(df[trm], src, glob_, loo=True)
va = bm.attach_features(df[~trm], src, glob_, loo=False)

# 節間モーター列を (date,venue,race,boat) のキーで直接マッピング（dtype非依存）
mkey = list(zip(df["date"].astype(str), df["venue"].astype(int),
                df["race"].astype(int), df["boat"].astype(int)))
mt = dict(zip(mkey, df["motor_setsu_top2"]))
mn = dict(zip(mkey, df["motor_setsu_n"]))
for frame in (tr, va):
    kk = list(zip(frame["date"].astype(str), frame["venue"].astype(int),
                  frame["race"].astype(int), frame["boat"].astype(int)))
    frame["motor_setsu_top2"] = [mt.get(x, np.nan) for x in kk]
    frame["motor_setsu_n"] = [mn.get(x, 0.0) for x in kk]
log(f"  結合確認: tr未マッチ={tr['motor_setsu_top2'].isna().sum()} / {len(tr)}")

EXTRA = ["motor_setsu_top2", "motor_setsu_n"]


RACE_KEYS = None
def train_eval(feats, seed):
    global RACE_KEYS
    k = int(len(tr) * 0.97)
    dtr = lgb.Dataset(tr.iloc[:k][feats], tr.iloc[:k]["target"],
                      categorical_feature=[c for c in CA if c in feats], free_raw_data=False)
    des = lgb.Dataset(tr.iloc[k:][feats], tr.iloc[k:]["target"], reference=dtr,
                      categorical_feature=[c for c in CA if c in feats], free_raw_data=False)
    pa = dict(bm.PARAMS); pa["verbose"] = -1; pa["seed"] = seed
    model = lgb.train(pa, dtr, 3000, valid_sets=[des], valid_names=["es"],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
    v = va.copy()
    v["pred"] = model.predict(v[feats], num_iteration=model.best_iteration)
    auc = roc_auc_score(v["target"], v["pred"])
    n = h = 0
    for (dt, ve, rc), gg in v.groupby(["date", "venue", "race"]):
        gg = gg.sort_values("boat")
        if len(gg) != 6 or not (gg["target"] == 1).any():
            continue
        top1 = int(gg["pred"].values.argmax()) + 1
        winner = int(gg[gg["target"] == 1]["boat"].iloc[0])
        n += 1; h += int(top1 == winner)
    return auc, h / n


log("\n=== 学習＆検証（2025以降・複数seedで平均）===")
SEEDS = [1, 7, 42]
base_a, base_acc, ext_a, ext_acc = [], [], [], []
for sd in SEEDS:
    a0, c0 = train_eval(FE, sd)
    a1, c1 = train_eval(FE + EXTRA, sd)
    base_a.append(a0); base_acc.append(c0); ext_a.append(a1); ext_acc.append(c1)
    log(f"  seed{sd:>2}: AUC {a0:.4f}→{a1:.4f} ({a1-a0:+.4f})  "
        f"的中 {c0*100:.2f}%→{c1*100:.2f}% ({(c1-c0)*100:+.2f}pt)")

import statistics as st
da = statistics_mean = sum(ext_a)/len(SEEDS) - sum(base_a)/len(SEEDS)
dacc = sum(ext_acc)/len(SEEDS) - sum(base_acc)/len(SEEDS)
log("\n=== 平均（3seed）===")
log(f"  AUC      : {sum(base_a)/3:.4f} → {sum(ext_a)/3:.4f}   （平均差 {da:+.4f}）")
log(f"  本命的中率: {sum(base_acc)/3*100:.2f}% → {sum(ext_acc)/3*100:.2f}%   （平均差 {dacc*100:+.2f}pt）")
allpos = all(ext_a[i] > base_a[i] for i in range(len(SEEDS)))
log(f"  全seedでプラス？ : {'はい' if allpos else 'いいえ（ばらつきあり）'}")

verdict = ("採用の価値あり（小さいが安定して正）" if (da >= 0.001 and allpos)
           else ("微妙（小さい/不安定）" if da > 0 else "効果なし → 採用しない"))
log(f"\n判定：{verdict}")
log("※本番モデルは未変更。複数seedで一貫してプラスなら本物の信号。")
