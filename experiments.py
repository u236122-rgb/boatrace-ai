# -*- coding: utf-8 -*-
"""
experiments.py  ―  予想モデル改善の検証スクリプト
------------------------------------------------------------
build_model.py と同じ「公正なLOO検証」(学習期間のみで実績テーブルを作り、
学習行は自分を除外(LOO)、検証行は素引き) で、以下を1つずつ足して
AUC と 1着的中率(本命1点) がどう動くかを比較する。

  1. 特徴量の交互作用（級別×当地、コース×持ちST 等）
  2. 選手×コースの相性をより細かく集計（3着内率/平均着順/経験本数/ST相対）
  3. ハイパーパラメータ調整（グリッドサーチ）
  4. 複数モデルのアンサンブル（LightGBM×複数 + ロジスティック回帰）

過学習に注意し、検証で本当に良くなったものだけ採用する。
"""
import sys, io, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import build_model as bm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
np.random.seed(0)


def log(m): print(m, flush=True)


# ============================================================
# データ準備（build_model と同じ前処理）
# ============================================================
log("データ読み込み・前処理 ...")
entries = pd.read_csv("data/csv/entries.csv")
results = pd.read_csv("data/csv/results.csv")
res = results[["boat", "date", "venue", "race", "course", "st",
               "finish", "weather", "wind_dir", "wind", "wave"]].copy()
df = entries.merge(res, on=["boat", "date", "venue", "race"], how="inner")
df = df.dropna(subset=["finish"]).reset_index(drop=True)
df["target"] = (df["finish"] == 1).astype(int)
df["win"] = df["target"]
df["top2"] = (df["finish"] <= 2).astype(int)
df["top3"] = (df["finish"] <= 3).astype(int)
df["st_num"] = pd.to_numeric(df["st"], errors="coerce")
df["fin_num"] = pd.to_numeric(df["finish"], errors="coerce")
df = bm.prep_common(df)
df = df.sort_values("date").reset_index(drop=True)
cut = int(len(df) * 0.9)
src_cols = ["racer_id", "course", "venue", "date", "win", "top2", "st_num"]
tr_raw, va_raw = df.iloc[:cut], df.iloc[cut:]
glob = bm.build_glob(tr_raw[src_cols])
src = tr_raw[src_cols]

log("ベース実績特徴を付与（LOO/素引き）...")
tr = bm.attach_features(tr_raw, src, glob, loo=True)
va = bm.attach_features(va_raw, src, glob, loo=False)
BASE = list(bm.FEATURES)
CATEG = list(bm.CATEG)
log(f"学習 {len(tr):,} / 検証 {len(va):,} 行 / 検証レース "
    f"{va.groupby(['date','venue','race']).ngroups:,}")


# ============================================================
# 共通：学習と評価
# ============================================================
def run_lgb(feats, params, seed=1, num_round=1500, es=60):
    p = dict(params); p["seed"] = seed; p["verbose"] = -1
    dtr = lgb.Dataset(tr[feats], tr["target"], categorical_feature=CATEG,
                      free_raw_data=False)
    dva = lgb.Dataset(va[feats], va["target"], reference=dtr,
                      categorical_feature=CATEG, free_raw_data=False)
    m = lgb.train(p, dtr, num_round, valid_sets=[dva], valid_names=["v"],
                  callbacks=[lgb.early_stopping(es, verbose=False)])
    pred = m.predict(va[feats], num_iteration=m.best_iteration)
    return pred, m.best_iteration


_va_keys = va[["date", "venue", "race", "target"]].reset_index(drop=True)


def metrics(pred):
    auc = roc_auc_score(va["target"], pred)
    t = _va_keys.copy(); t["p"] = np.asarray(pred)
    hit = t.sort_values("p").groupby(["date", "venue", "race"]).tail(1)["target"].mean()
    return auc, hit


RESULTS = []
def report(name, pred, extra=""):
    auc, hit = metrics(pred)
    RESULTS.append((name, auc, hit))
    log(f"  {name:<34} AUC={auc:.4f}  1着的中={hit*100:5.2f}%  {extra}")
    return auc, hit


# ============================================================
# 0. ベースライン
# ============================================================
log("\n=== 0. ベースライン（現行モデル）===")
base_params = dict(bm.PARAMS)
t0 = time.time()
pred_base, it = run_lgb(BASE, base_params)
auc_b, hit_b = report(f"baseline (木{it})", pred_base)
log(f"  (学習 {time.time()-t0:.0f}秒)")


# ============================================================
# 1. 交互作用特徴
# ============================================================
log("\n=== 1. 交互作用特徴を追加 ===")
def add_inter(d):
    d["ix_grade_rvwin"] = d["grade_num"] * d["rv_win"]      # 級別×当地勝率
    d["ix_grade_winnat"] = d["grade_num"] * d["win_nat"]    # 級別×全国勝率
    d["ix_course_rcst"] = d["boat"] * d["rc_st"]            # コース×持ちST
    d["ix_winnat_rcwin"] = d["win_nat"] * d["rc_win"]       # 実力×コース適性
    d["ix_motor_rcwin"] = d["motor_top2"] * d["rc_win"]     # 機力×コース適性
    d["ix_wave_boat"] = d["wave"] * d["boat"]               # 波×枠
    d["ix_wind_boat"] = d["wind"] * d["boat"]               # 風×枠
add_inter(tr); add_inter(va)
INTER = ["ix_grade_rvwin", "ix_grade_winnat", "ix_course_rcst",
         "ix_winnat_rcwin", "ix_motor_rcwin", "ix_wave_boat", "ix_wind_boat"]
pred1, it1 = run_lgb(BASE + INTER, base_params)
report(f"+交互作用 (木{it1})", pred1)


# ============================================================
# 2. 選手×コースの相性をより細かく集計
# ============================================================
log("\n=== 2. 選手×コース集計を細かく ===")
def loo_mean(target, source, keys, col, prior, k, loo):
    s, c = bm._grp_sumcount(source, keys, col)
    tmp = target[keys].merge(s, on=keys, how="left").merge(c, on=keys, how="left")
    sv = tmp["s"].values.astype(float); cv = tmp["c"].values.astype(float)
    if loo:
        own = target[col].values.astype(float)
        valid = ~np.isnan(own)
        sv = np.where(np.isnan(sv), 0.0, sv) - np.where(valid, np.nan_to_num(own), 0.0)
        cv = np.where(np.isnan(cv), 0.0, cv) - valid.astype(float)
    else:
        sv = np.nan_to_num(sv); cv = np.nan_to_num(cv)
    pri = prior if np.isscalar(prior) else prior
    return (sv + pri * k) / (cv + k), cv

src2 = tr_raw[["racer_id", "course", "top3", "fin_num", "st_num"]]
c_top3 = tr_raw.groupby("course")["top3"].mean().to_dict()
c_fin = tr_raw.groupby("course")["fin_num"].mean().to_dict()
g_top3 = float(tr_raw["top3"].mean()); g_fin = float(tr_raw["fin_num"].mean())
for d, raw, loo in ((tr, tr_raw, True), (va, va_raw, False)):
    keyc = ["racer_id", "course"]
    base_keys = raw[keyc + ["top3", "fin_num", "st_num"]]
    v, n = loo_mean(base_keys, src2, keyc, "top3",
                    raw["course"].map(c_top3).fillna(g_top3).values, bm.K_COURSE, loo)
    d["rc_top3"] = v
    v2, _ = loo_mean(base_keys, src2, keyc, "fin_num",
                     raw["course"].map(c_fin).fillna(g_fin).values, bm.K_COURSE, loo)
    d["rc_avgfin"] = v2
    d["rc_n"] = np.log1p(n)
# レース内でのコース別STの相対（速い順）
for d in (tr, va):
    d["rc_st_rel"] = d["rc_st"] - d.groupby(["date", "venue", "race"])["rc_st"].transform("mean")
FINER = ["rc_top3", "rc_avgfin", "rc_n", "rc_st_rel"]
pred2, it2 = run_lgb(BASE + FINER, base_params)
report(f"+細かい選手×コース (木{it2})", pred2)

# 1+2 両方
pred12, it12 = run_lgb(BASE + INTER + FINER, base_params)
report(f"+交互作用+細かい集計 (木{it12})", pred12)


# ============================================================
# 3. ハイパーパラメータ・グリッドサーチ
# ============================================================
log("\n=== 3. ハイパーパラメータ探索（特徴=BASE+FINER）===")
FEAT3 = BASE + FINER
grid = [
    dict(num_leaves=31,  min_data_in_leaf=500,  learning_rate=0.05, lambda_l2=5),
    dict(num_leaves=63,  min_data_in_leaf=500,  learning_rate=0.05, lambda_l2=5),   # ≒現行
    dict(num_leaves=63,  min_data_in_leaf=1000, learning_rate=0.05, lambda_l2=10),
    dict(num_leaves=127, min_data_in_leaf=1000, learning_rate=0.05, lambda_l2=10),
    dict(num_leaves=127, min_data_in_leaf=2000, learning_rate=0.03, lambda_l2=10),
    dict(num_leaves=255, min_data_in_leaf=2000, learning_rate=0.03, lambda_l2=20),
    dict(num_leaves=95,  min_data_in_leaf=1500, learning_rate=0.03, lambda_l2=15),
    dict(num_leaves=63,  min_data_in_leaf=800,  learning_rate=0.03, lambda_l2=8),
]
best = None
for gp in grid:
    p = dict(base_params); p.update(gp)
    p["feature_fraction"] = 0.8; p["bagging_fraction"] = 0.8; p["bagging_freq"] = 1
    pred, it = run_lgb(FEAT3, p, num_round=2500, es=80)
    auc, hit = report(f"leaves{gp['num_leaves']} minleaf{gp['min_data_in_leaf']} "
                      f"lr{gp['learning_rate']} l2{gp['lambda_l2']} (木{it})", pred)
    if best is None or auc > best[1]:
        best = (gp, auc, hit, p, pred)
log(f"  → 最良: leaves{best[0]['num_leaves']} minleaf{best[0]['min_data_in_leaf']} "
    f"lr{best[0]['learning_rate']} l2{best[0]['lambda_l2']}  AUC={best[1]:.4f}")
best_params = best[3]


# ============================================================
# 4. アンサンブル
# ============================================================
log("\n=== 4. アンサンブル ===")
# LightGBM をシード替え/別パラメータで複数
predA, _ = run_lgb(FEAT3, best_params, seed=1, num_round=2500, es=80)
predB, _ = run_lgb(FEAT3, best_params, seed=7, num_round=2500, es=80)
alt = dict(best_params); alt.update(num_leaves=255, min_data_in_leaf=2000,
                                    learning_rate=0.02, lambda_l2=20)
predC, _ = run_lgb(FEAT3, alt, seed=3, num_round=4000, es=100)

# ロジスティック回帰（多様性のため）
num_feats = [f for f in FEAT3 if f not in CATEG]
sc = StandardScaler()
Xtr = sc.fit_transform(np.nan_to_num(tr[num_feats].values))
Xva = sc.transform(np.nan_to_num(va[num_feats].values))
lr = LogisticRegression(max_iter=2000, C=1.0)
lr.fit(Xtr, tr["target"].values)
predL = lr.predict_proba(Xva)[:, 1]
report("ロジスティック回帰 単体", predL)

def rank_avg(preds):
    r = np.zeros(len(preds[0]))
    for p in preds:
        r += pd.Series(p).rank(pct=True).values
    return r / len(preds)

report("LGB×2 (seed平均)", rank_avg([predA, predB]))
report("LGB×3 (seed/別param)", rank_avg([predA, predB, predC]))
report("LGB×3 + ロジスティック", rank_avg([predA, predB, predC, predL]))
# ロジの重みを下げた版（LGB3 : ロジ = 3 : 0.5）
rL = pd.Series(predL).rank(pct=True).values
report("LGB×3 + ロジ(重み0.5)",
       (3 * rank_avg([predA, predB, predC]) + 0.5 * rL) / 3.5)


# ============================================================
# まとめ
# ============================================================
log("\n=================  まとめ（AUC降順）=================")
RESULTS.sort(key=lambda x: -x[1])
log(f"  {'手法':<34} {'AUC':>8} {'1着的中率':>10}  (baseline比)")
for name, auc, hit in RESULTS:
    log(f"  {name:<34} {auc:8.4f} {hit*100:9.2f}%   "
        f"AUC{auc-auc_b:+.4f} / 的中{ (hit-hit_b)*100:+.2f}pt")
log("\n※ 検証レース約3万→1着的中率の誤差±0.3%程度。AUCは±0.001でも有意。")
