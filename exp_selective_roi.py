# -*- coding: utf-8 -*-
"""
exp_selective_roi.py ― 条件を絞ると控除率(25%)を超えられるか検証
  事前情報だけで選別（自信conf・場の荒れ度・風速・モデルの想定配当帯）。
  券種：単勝(本命1点)・2連複(上位3点)・3連単(上位3点) を中心に、
  「全部買う」vs「条件で絞る」の回収率を比較。
  2024末まで学習→2025以降で検証（リーク無し／本番モデル不変）。
"""
import sys, io
import numpy as np
import pandas as pd
import lightgbm as lgb
import build_model as bm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def log(m): print(m, flush=True)


def bet_tops(s):
    S = s.sum()
    tri = {}
    for a in range(6):
        for b in range(6):
            if b == a:
                continue
            for c in range(6):
                if c in (a, b):
                    continue
                tri[(a+1, b+1, c+1)] = s[a]/S * s[b]/(S-s[a]) * s[c]/(S-s[a]-s[b])
    exa = {}
    for a in range(6):
        den = S - s[a]
        for b in range(6):
            if b != a:
                exa[(a+1, b+1)] = s[a]/S * s[b]/den
    quin = {}
    for a in range(6):
        for b in range(a+1, 6):
            quin[(a+1, b+1)] = exa[(a+1, b+1)] + exa[(b+1, a+1)]

    def t3(d):
        return ["-".join(map(str, k)) for k, _ in sorted(d.items(), key=lambda x: -x[1])[:3]]
    top_tri = sorted(tri.items(), key=lambda x: -x[1])[0]
    return t3(tri), t3(quin), top_tri          # 3連単top3, 2連複top3, 3連単最有力(組,確率)


# ---- 学習 ----
log("学習（2024末まで・節間モーター込み）...")
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
trm = df["date"] < "2025-01-01"
sc = ["racer_id", "course", "venue", "date", "win", "top2", "st_num"]
glob_ = bm.build_glob(df[trm][sc])
FE = [f for f in bm.FEATURES if "exhibit" not in f]
CA = [c for c in bm.CATEG if c in FE]
tr = bm.attach_features(df[trm], df[trm][sc], glob_, loo=True)
va = bm.attach_features(df[~trm], df[trm][sc], glob_, loo=False)
key = lambda f: list(zip(f["date"].astype(str), f["venue"].astype(int),
                         f["race"].astype(int), f["boat"].astype(int)))
mt = dict(zip(key(df), df["motor_setsu_top2"])); mn = dict(zip(key(df), df["motor_setsu_n"]))
for f in (tr, va):
    f["motor_setsu_top2"] = [mt[k] for k in key(f)]
    f["motor_setsu_n"] = [mn[k] for k in key(f)]
k = int(len(tr) * 0.97)
dtr = lgb.Dataset(tr.iloc[:k][FE], tr.iloc[:k]["target"], categorical_feature=CA, free_raw_data=False)
des = lgb.Dataset(tr.iloc[k:][FE], tr.iloc[k:]["target"], reference=dtr, categorical_feature=CA, free_raw_data=False)
pa = dict(bm.PARAMS); pa["verbose"] = -1
model = lgb.train(pa, dtr, 3000, valid_sets=[des], valid_names=["es"],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
va = va.copy(); va["pred"] = model.predict(va[FE], num_iteration=model.best_iteration)

# ---- 実払戻 ----
log("実払戻＋場の荒れ度を準備 ...")
pall = pd.read_csv("data/csv/payouts_all.csv")
pall = pall[pall["date"] >= "2025-01-01"]
W = {bt: {} for bt in ("単勝", "2連複", "3連単")}
for date, v, r, bt, combo, pay in zip(pall["date"], pall["venue"], pall["race"],
                                      pall["bet_type"], pall["combo"], pall["payout"]):
    if bt in W:
        W[bt][(date, int(v), int(r))] = (combo, int(pay))
# 場の荒れ度（まくり系率）
km = pd.read_csv("data/csv/kimarite.csv")
km = km[km["kimarite"].notna()]
km["are"] = km["kimarite"].isin(["まくり", "まくり差し"]).astype(int)
arate = km.groupby("venue")["are"].mean().to_dict()

# ---- レース単位の記録 ----
log("レース別に記録を作成 ...")
rec = []
for (dt, v, r), g in va.groupby(["date", "venue", "race"]):
    g = g.sort_values("boat")
    if len(g) != 6:
        continue
    raw = np.clip(g["pred"].values, 1e-9, None)
    s = raw / raw.sum()
    tri3, quin3, top_tri = bet_tops(s)
    topboat = int(np.argmax(s)) + 1
    conf = float(s.max())
    est_pay = int(round(0.75 / max(top_tri[1], 1e-9) * 100))   # 最有力3連単の想定配当
    wind = pd.to_numeric(g["wind"].iloc[0], errors="coerce")
    rowd = {"date": dt, "venue": int(v), "race": int(r), "conf": conf,
            "wind": float(wind) if pd.notna(wind) else 0.0,
            "arate": arate.get(int(v), 0.27), "est_pay": est_pay}
    for bt, pick in (("単勝", str(topboat)), ("2連複", quin3), ("3連単", tri3)):
        w = W[bt].get((dt, int(v), int(r)))
        if w is None:
            rowd[bt+"_n"] = 0; rowd[bt+"_pay"] = 0; continue
        hit = (w[0] == pick) if bt == "単勝" else (w[0] in pick)
        rowd[bt+"_n"] = 1
        rowd[bt+"_pay"] = w[1] if hit else 0
        rowd[bt+"_hit"] = int(hit)
    rec.append(rowd)
R = pd.DataFrame(rec)


def show(name, sub, bt, pts):
    s = sub[sub[bt+"_n"] == 1]
    n = len(s)
    if n < 50:
        log(f"  {name:<26} 対象{n:>6}R  （少なすぎ）"); return
    cost = n * pts * 100
    ret = s[bt+"_pay"].sum()
    hit = s[bt+"_hit"].mean() * 100
    flag = "  ★100%超!" if ret/cost*100 >= 100 else ""
    log(f"  {name:<26} 対象{n:>6}R  的中{hit:>5.1f}%  回収{ret/cost*100:>6.1f}%{flag}")


log("\n========== 単勝（本命1点＝モデル最有力艇）==========")
show("全レース", R, "単勝", 1)
for lo, hi in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.75),
               (0.75, 0.80), (0.80, 0.85), (0.85, 1.1)]:
    show(f"自信 {lo:.2f}〜{hi:.2f}", R[(R.conf >= lo) & (R.conf < hi)], "単勝", 1)
log("  -- 堅い場・極限本命との合わせ --")
show("堅い場(荒れ度<25%)×自信0.6+", R[(R.arate < 0.25) & (R.conf >= 0.6)], "単勝", 1)
show("堅い場×自信0.75+", R[(R.arate < 0.25) & (R.conf >= 0.75)], "単勝", 1)
show("自信0.80+ かつ 想定配当 本命堅", R[(R.conf >= 0.80) & (R.est_pay < 600)], "単勝", 1)

log("\n========== 2連複（上位3点）==========")
show("全レース", R, "2連複", 3)
for lo, hi in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.1)]:
    show(f"自信 {lo:.1f}〜{hi:.1f}", R[(R.conf >= lo) & (R.conf < hi)], "2連複", 3)
show("堅い場×自信0.55+", R[(R.arate < 0.25) & (R.conf >= 0.55)], "2連複", 3)
show("荒れ場(>=30%)×風5m+", R[(R.arate >= 0.30) & (R.wind >= 5)], "2連複", 3)

log("\n========== 3連単（上位3点）==========")
show("全レース", R, "3連単", 3)
for lo, hi in [(0.0, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.1)]:
    show(f"自信 {lo:.2f}〜{hi:.2f}", R[(R.conf >= lo) & (R.conf < hi)], "3連単", 3)
log("\n  -- 3連単・想定配当帯別（最有力1点の配当でレースを分類）--")
for lo, hi, lab in [(0, 800, "本命堅(〜800円)"), (800, 1500, "800〜1500"),
                    (1500, 3000, "1500〜3000"), (3000, 9e9, "3000円〜(荒れ)")]:
    show(lab, R[(R.est_pay >= lo) & (R.est_pay < hi)], "3連単", 3)

log("\n※基準線：控除率25%＝回収率75%。100%超なら理論上プラス。")
log("※選別はすべて事前情報（自信・場・風・想定配当）。決まり手は結果論なので不使用。")
log("※リーク無し（2024末まで学習→2025検証）。本番モデル不変。")
