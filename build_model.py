# -*- coding: utf-8 -*-
"""
build_model.py  ―  競艇 1着予想モデルの学習スクリプト（強化版）
------------------------------------------------------------
data/csv/entries.csv（出走表）と data/csv/results.csv（結果）を結合し、
LightGBM で「その艇が1着になる確率」を学習する。

★強化ポイント（登録番号から自動で引く実績データ）
  ・コース別実績  … その選手が1〜6コースそれぞれで勝つ率/2連率/平均ST
  ・当地実績      … その場(venue)での勝率/2連率
  ・スタート力    … 通算平均ST
  ・直近の調子    … 直近20走の勝率
  ・モーター      … 直近のモーター2連率（自動取得）
  少ない標本はベイズ収縮で平均へ寄せてノイズを抑える。

データを更新したら、このスクリプトを実行し直すだけでモデルが作り直される。
    python build_model.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "data", "csv")
OUT = os.path.join(BASE, "model")
os.makedirs(OUT, exist_ok=True)

GRADE_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
WEATHER_LIST = ["晴", "曇り", "雨", "雪", "霧"]
WINDDIR_LIST = ["無風", "北", "北東", "東", "南東", "南", "南西", "西", "北西"]
VENUE_NAMES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}

# 収縮の強さ（標本がこの数あれば実測:平均=1:1）
K_COURSE = 12
K_VENUE = 8
K_RECENT = 0  # 直近は収縮なし

FEATURES = [
    "boat",            # 枠番（=進入コースの代理）
    "venue",
    "grade_num", "age", "weight",
    "win_nat", "top2_nat", "win_loc", "top2_loc",
    "motor_top2", "boat_top2",
    "weather_code", "winddir_code", "wind", "wave",
    # 相対特徴
    "win_nat_rel", "top2_nat_rel", "motor_rel",
    # ★追加：選手の実績（登録番号から自動取得・Leave-One-Outでリーク防止）
    "rc_win", "rc_top2", "rc_st",   # コース別 勝率/2連率/平均ST
    "rv_win", "rv_top2",            # 当地 勝率/2連率
    "r_avg_st",                     # 通算平均ST
    "rc_win_rel",                   # コース別勝率のレース内相対
    # ★展示タイム（直前情報・任意入力。無い時は中立値で劣化なし）
    "exhibit_f", "exhibit_rel",
    # ★節間モーター成績（その節の前日までの2連率＝仕上がり。検証で+0.25pt）
    "motor_setsu_top2", "motor_setsu_n",
]
CATEG = ["venue", "weather_code", "winddir_code"]


def log(msg):
    print("[build] " + msg, flush=True)


def shrink(s, c, prior, k):
    """ベイズ収縮：合計 s / 件数 c の標本平均を prior に向けて k で寄せる。"""
    return (s + prior * k) / (c + k)


def build_glob(m):
    """コース別/全体の事前分布（収縮先・欠損補完用）。"""
    return {
        "course_win": m.groupby("course")["win"].mean().to_dict(),
        "course_top2": m.groupby("course")["top2"].mean().to_dict(),
        "course_st": m.groupby("course")["st_num"].mean().to_dict(),
        "g_win": float(m["win"].mean()),
        "g_top2": float(m["top2"].mean()),
        "g_st": float(m["st_num"].mean(skipna=True)),
    }


def _grp_sumcount(src, keys, col):
    """src を keys で集計し col の合計・件数(非NaN)を返す。"""
    g = src.groupby(keys)[col]
    return g.sum().rename("s"), g.count().rename("c")


def attach_features(target, src, glob, loo):
    """target 各行に、src から作った実績特徴を付与。
    loo=True のとき（src と target が同一集合）は自分の結果を除外して計算
    （自己リーク防止）。loo=False は src 値をそのまま参照（=未来予想と同じ）。"""
    df = target.copy()

    def feat(keys, col, prior_series, k, outname):
        s, c = _grp_sumcount(src, keys, col)
        tmp = df.merge(s, on=keys, how="left").merge(c, on=keys, how="left")
        sv = tmp["s"].values
        cv = tmp["c"].values
        if loo:
            own = df[col].values
            valid = ~np.isnan(own)
            sv = np.where(np.isnan(sv), 0.0, sv) - np.where(valid, np.nan_to_num(own), 0.0)
            cv = np.where(np.isnan(cv), 0.0, cv) - valid.astype(float)
        else:
            sv = np.where(np.isnan(sv), 0.0, sv)
            cv = np.where(np.isnan(cv), 0.0, cv)
        prior = df[prior_series[0]].map(prior_series[1]).values \
            if prior_series[0] else np.full(len(df), prior_series[1])
        out = (sv + prior * k) / (cv + k)
        df[outname] = out

    feat(["racer_id", "course"], "win", ("course", glob["course_win"]), K_COURSE, "rc_win")
    feat(["racer_id", "course"], "top2", ("course", glob["course_top2"]), K_COURSE, "rc_top2")
    feat(["racer_id", "course"], "st_num", ("course", glob["course_st"]), K_COURSE, "rc_st")
    feat(["racer_id", "venue"], "win", (None, glob["g_win"]), K_VENUE, "rv_win")
    feat(["racer_id", "venue"], "top2", (None, glob["g_top2"]), K_VENUE, "rv_top2")
    feat(["racer_id"], "st_num", (None, glob["g_st"]), 6, "r_avg_st")

    grp = df.groupby(["date", "venue", "race"])
    df["rc_win_rel"] = df["rc_win"] - grp["rc_win"].transform("mean")
    return df


def prep_common(df):
    """共通の数値化・コード化・相対特徴。"""
    df["grade_num"] = df["grade"].map(GRADE_MAP).fillna(2).astype(int)
    df["weather_code"] = df["weather"].map(
        {w: i for i, w in enumerate(WEATHER_LIST)}).fillna(1).astype(int)
    df["winddir_code"] = df["wind_dir"].map(
        {w: i for i, w in enumerate(WINDDIR_LIST)}).fillna(0).astype(int)
    for c in ["age", "weight", "win_nat", "top2_nat", "win_loc", "top2_loc",
              "motor_top2", "boat_top2", "wind", "wave"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    grp = df.groupby(["date", "venue", "race"])
    df["win_nat_rel"] = df["win_nat"] - grp["win_nat"].transform("mean")
    df["top2_nat_rel"] = df["top2_nat"] - grp["top2_nat"].transform("mean")
    df["motor_rel"] = df["motor_top2"] - grp["motor_top2"].transform("mean")
    # 展示タイム（あれば）。欠損は中央値で埋め、レース内相対も作る。
    if "exhibit" in df.columns:
        df["exhibit"] = pd.to_numeric(df["exhibit"], errors="coerce")
        med = df["exhibit"].median()
        df["exhibit_f"] = df["exhibit"].fillna(med)
        df["exhibit_rel"] = df["exhibit_f"] - grp["exhibit_f"].transform("mean")
    return df


def add_motor_setsu(df):
    """『節間モーター成績』を列追加。
      その節（=同一場の連続開催。3日以上空けば別節）の“前日まで”の
      モーター2連率(motor_setsu_top2, 0-1)と試走数(motor_setsu_n)。
      当日の結果は使わない＝リーク無し。節初日(実績なし)は番組表の
      モーター2連率(%→比率)で代用。core.py の予想時計算と同じ定義。
      df は entries+results 結合済み（motor_no, top2, date, venue を含む）。"""
    ser = {}
    for v, gv in df.groupby("venue"):
        sid, prev = 0, None
        for dd in sorted(gv["date"].unique()):
            if prev is not None and (pd.to_datetime(dd) - pd.to_datetime(prev)).days > 3:
                sid += 1
            ser[(v, dd)] = sid
            prev = dd
    series = [ser[(v, dd)] for v, dd in zip(df["venue"], df["date"])]
    k = pd.DataFrame({"venue": df["venue"].values, "series": series,
                      "motor_no": df["motor_no"].values, "date": df["date"].values,
                      "top2": df["top2"].values})
    daily = (k.groupby(["venue", "series", "motor_no", "date"])
             .agg(d_sum=("top2", "sum"), d_cnt=("top2", "size")).reset_index()
             .sort_values(["venue", "series", "motor_no", "date"]))
    g = daily.groupby(["venue", "series", "motor_no"])
    daily["prior_sum"] = g["d_sum"].cumsum() - daily["d_sum"]
    daily["prior_cnt"] = g["d_cnt"].cumsum() - daily["d_cnt"]
    daily["m_setsu"] = daily["prior_sum"] / daily["prior_cnt"]
    mp = {(int(v), int(s), int(m), dt): (sc, int(cnt))
          for v, s, m, dt, sc, cnt in zip(daily["venue"], daily["series"],
              daily["motor_no"], daily["date"], daily["m_setsu"], daily["prior_cnt"])}
    setsu, ncol = [], []
    for v, s, m, dt, mt in zip(df["venue"], series, df["motor_no"], df["date"],
                               df["motor_top2"]):
        rec = mp.get((int(v), int(s), int(m), dt))
        if rec is None or rec[1] == 0 or pd.isna(rec[0]):
            setsu.append(float(mt) / 100.0 if pd.notna(mt) else np.nan)
            ncol.append(0.0)
        else:
            setsu.append(float(rec[0])); ncol.append(float(rec[1]))
    out = df.copy()
    out["motor_setsu_top2"] = setsu
    out["motor_setsu_top2"] = out["motor_setsu_top2"].fillna(
        df["motor_top2"].mean() / 100.0)
    out["motor_setsu_n"] = ncol
    return out


# パラメータ（小・強正則）。exp_simplify.py の「未来の年まるごと」検証で、
# 複雑な旧設定より単純・強正則の方が一貫して上回った：
#   的中率 2024 +1.30pt / 2025 +1.02pt（両年・全seedで勝ち、平均55.8%→57.0%）。
# 旧設定は過学習気味で、未来データ(ドリフト)に弱かったため置き換え。
# --- 旧設定（複雑・いつでも戻せるよう保存）---
# PARAMS = {
#     "objective": "binary", "metric": ["auc", "binary_logloss"],
#     "learning_rate": 0.03, "num_leaves": 127, "min_data_in_leaf": 2000,
#     "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
#     "lambda_l2": 10.0, "verbose": -1,
# }
PARAMS = {
    "objective": "binary", "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 10000,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 30.0, "verbose": -1,
}


def fit(tr, va, num_round=2500, es=80, verbose=True):
    dtr = lgb.Dataset(tr[FEATURES], label=tr["target"], categorical_feature=CATEG)
    dva = lgb.Dataset(va[FEATURES], label=va["target"], categorical_feature=CATEG,
                      reference=dtr)
    cbs = [lgb.early_stopping(es)]
    if verbose:
        cbs.append(lgb.log_evaluation(100))
    model = lgb.train(PARAMS, dtr, num_boost_round=num_round,
                      valid_sets=[dtr, dva], valid_names=["train", "valid"],
                      callbacks=cbs)
    va = va.copy()
    va["pred"] = model.predict(va[FEATURES], num_iteration=model.best_iteration)
    hit = va.sort_values("pred").groupby(["date", "venue", "race"]).tail(1)
    return model, float(hit["target"].mean()), model.best_iteration


def main():
    log("CSV 読み込み中 ...")
    entries = pd.read_csv(os.path.join(CSV, "entries.csv"))
    results = pd.read_csv(os.path.join(CSV, "results.csv"))
    log(f"entries={len(entries):,} 行 / results={len(results):,} 行")

    res = results[["boat", "date", "venue", "race", "course", "st",
                   "finish", "weather", "wind_dir", "wind", "wave",
                   "exhibit"]].copy()
    df = entries.merge(res, on=["boat", "date", "venue", "race"], how="inner")
    df = df.dropna(subset=["finish"]).reset_index(drop=True)
    df["target"] = (df["finish"] == 1).astype(int)
    df["win"] = df["target"]
    df["top2"] = (df["finish"] <= 2).astype(int)
    df["st_num"] = pd.to_numeric(df["st"], errors="coerce")
    log(f"結合後 {len(df):,} 行")

    df = prep_common(df)
    df = df.sort_values("date").reset_index(drop=True)
    log("節間モーター成績を付与 ...")
    df = add_motor_setsu(df)
    cut_idx = int(len(df) * 0.9)
    src_cols = ["racer_id", "course", "venue", "date", "win", "top2", "st_num"]

    # ---- 検証用：学習期間のみで実績を作り、公正に評価 ----
    #   学習行=LOO（自分除外）、検証行=学習テーブル参照（=未来予想と同じ・リーク無し）
    log("実績特徴を付与（学習期間ソース）→ 公正な検証 ...")
    tr_raw = df.iloc[:cut_idx]
    va_raw = df.iloc[cut_idx:]
    glob_t = build_glob(tr_raw[src_cols])
    src_t = tr_raw[src_cols]
    tr_eval = attach_features(tr_raw, src_t, glob_t, loo=True)
    va_eval = attach_features(va_raw, src_t, glob_t, loo=False)
    _, acc, _ = fit(tr_eval, va_eval, verbose=True)
    log(f"検証 1着的中率（本命1点・リーク無し）= {acc*100:.1f}%")

    # ---- 本番：全データをソースに（学習はLOO）、最終モデル ----
    log("実績特徴を付与（全データソース）→ 最終モデル学習 ...")
    glob_a = build_glob(df[src_cols])
    src_a = df[src_cols]
    df_all = attach_features(df, src_a, glob_a, loo=True)
    fin_cut = int(len(df_all) * 0.9)
    model, _, best_it = fit(df_all.iloc[:fin_cut], df_all.iloc[fin_cut:],
                            verbose=False)

    # ---- 選手プロフィール（最新の出走データから）----
    log("選手プロフィール作成中 ...")
    ent = entries.copy()
    ent["grade_num"] = ent["grade"].map(GRADE_MAP).fillna(2).astype(int)
    ent = ent.sort_values("date")
    racer_global = (ent.groupby("racer_id").tail(1)
                    [["racer_id", "name", "grade_num", "age", "weight",
                      "win_nat", "top2_nat", "win_loc", "top2_loc"]]
                    .set_index("racer_id"))
    # モーター：直近20走の2連率平均（自動取得）
    motor_recent = (ent.groupby("racer_id").tail(20)
                    .groupby("racer_id")["motor_top2"].mean())
    boat_recent = (ent.groupby("racer_id").tail(20)
                   .groupby("racer_id")["boat_top2"].mean())
    racer_global["motor_top2"] = motor_recent
    racer_global["boat_top2"] = boat_recent
    racer_global["motor_top2"] = racer_global["motor_top2"].fillna(ent["motor_top2"].mean())
    racer_global["boat_top2"] = racer_global["boat_top2"].fillna(ent["boat_top2"].mean())

    # 場×コースの基礎1着率（参考表示用）
    rr_disp = results.dropna(subset=["finish"]).copy()
    rr_disp["win"] = (rr_disp["finish"] == 1).astype(int)
    venue_course = rr_disp.groupby(["venue", "course"])["win"].mean().unstack().fillna(0)

    # 進入予想用：場×枠 → 実際の進入コース 確率行列（6x6, 行=枠, 列=コース）
    log("進入(前づけ)傾向テーブル作成中 ...")
    fc = results.dropna(subset=["course"]).copy()
    frame_course = {}
    for v, d in fc.groupby("venue"):
        mat = (d.groupby(["boat", "course"]).size()
               .unstack(fill_value=0).reindex(index=range(1, 7),
                                              columns=range(1, 7), fill_value=0))
        mat = mat.values.astype(float)
        rowsum = mat.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        frame_course[int(v)] = (mat / rowsum).tolist()

    # 選手×枠 → 進入コース傾向（前づけは選手の癖。20走以上を採用）
    racer_frame_course = {}
    cnt = fc.groupby(["racer_id", "boat", "course"]).size()
    tot = fc.groupby(["racer_id", "boat"]).size()
    for (rid, boat), c in cnt.groupby(level=[0, 1]):
        n = int(tot.loc[(rid, boat)])
        if n < 20:
            continue
        row = np.zeros(6)
        for (_, _, course), v2 in c.items():
            row[int(course) - 1] = v2
        racer_frame_course[(int(rid), int(boat))] = (row / row.sum()).tolist()

    # ---- GUI 用の実績テーブル（全データ・収縮あり）----
    log("GUI用 実績テーブル作成中 ...")

    def shrunk_table(keys, col, prior_map, prior_const, k):
        g = src_a.groupby(keys)[col]
        s, c = g.sum(), g.count()
        out = pd.DataFrame({"s": s, "c": c}).reset_index()
        if prior_map is not None:
            pk = keys[-1]
            prior = out[pk].map(prior_map).values
        else:
            prior = np.full(len(out), prior_const)
        out["v"] = (out["s"] + prior * k) / (out["c"] + k)
        return out.set_index(keys)["v"]

    rc_win_s = shrunk_table(["racer_id", "course"], "win", glob_a["course_win"], None, K_COURSE)
    rc_top2_s = shrunk_table(["racer_id", "course"], "top2", glob_a["course_top2"], None, K_COURSE)
    rc_st_s = shrunk_table(["racer_id", "course"], "st_num", glob_a["course_st"], None, K_COURSE)
    rv_win_s = shrunk_table(["racer_id", "venue"], "win", None, glob_a["g_win"], K_VENUE)
    rv_top2_s = shrunk_table(["racer_id", "venue"], "top2", None, glob_a["g_top2"], K_VENUE)
    r_st_s = shrunk_table(["racer_id"], "st_num", None, glob_a["g_st"], 6)
    recent_s = (df.sort_values("date").groupby("racer_id").tail(20)
                .groupby("racer_id")["win"].mean())

    rc_dict = {k: (float(rc_win_s.get(k, np.nan)),
                   float(rc_top2_s.get(k, np.nan)),
                   float(rc_st_s.get(k, np.nan))) for k in rc_win_s.index}
    rv_dict = {k: (float(rv_win_s.get(k, np.nan)),
                   float(rv_top2_s.get(k, np.nan))) for k in rv_win_s.index}
    rr_dict = {int(k): float(v) for k, v in r_st_s.items()}
    rec_dict = {int(k): float(v) for k, v in recent_s.items()}

    defaults = {
        "grade_num": 2, "age": float(ent["age"].median()),
        "weight": float(ent["weight"].median()),
        "win_nat": float(ent["win_nat"].median()),
        "top2_nat": float(ent["top2_nat"].median()),
        "win_loc": float(ent["win_loc"].median()),
        "top2_loc": float(ent["top2_loc"].median()),
        "motor_top2": float(ent["motor_top2"].mean()),
        "boat_top2": float(ent["boat_top2"].mean()),
        # 展示タイムの中央値（手入力が無い時の中立値）
        "exhibit_f": float(pd.to_numeric(results["exhibit"], errors="coerce").median()),
    }

    log("保存中 ...")
    model.save_model(os.path.join(OUT, "lgb_model.txt"))
    joblib.dump({
        "features": FEATURES, "categ": CATEG, "grade_map": GRADE_MAP,
        "weather_list": WEATHER_LIST, "winddir_list": WINDDIR_LIST,
        "venue_names": VENUE_NAMES, "defaults": defaults,
        "racer_global": racer_global, "venue_course": venue_course,
        "frame_course": frame_course, "racer_frame_course": racer_frame_course,
        "rc_table": rc_dict, "rv_table": rv_dict,
        "rr_avg_st": rr_dict, "recent_win": rec_dict,
        "glob": glob_a,
        "valid_acc": float(acc),
        "data_max_date": str(entries["date"].max()),
        "data_min_date": str(entries["date"].min()),
        "n_races": int(entries.groupby(["date", "venue", "race"]).ngroups),
        "best_iteration": int(best_it),
    }, os.path.join(OUT, "tables.pkl"))
    log(f"完了！ model/ に保存（検証的中率 {acc*100:.1f}% / 木 {best_it}本）")


if __name__ == "__main__":
    main()
