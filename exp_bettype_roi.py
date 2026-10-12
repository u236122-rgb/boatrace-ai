# -*- coding: utf-8 -*-
"""
exp_bettype_roi.py ― A. 券種別おすすめの「的中率・回収率」を実払戻で検証
  ・2024末まで学習 → 2025以降で検証（リーク無し／本番モデルは不変）
  ・各レースで 3連単/3連複/2連単/2連複 の上位3点を算出
  ・payouts_all.csv の実際の当たり目・払戻と突き合わせ、券種ごとに
      的中率（上位3点に当たり目が入る率）／回収率（上位3点を各100円買い）
    を集計。さらに「おすすめ券種ロジック（自信で出し分け）」も検証。
"""
import sys, io
import numpy as np
import pandas as pd
import lightgbm as lgb
import build_model as bm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def log(m): print(m, flush=True)


def bet_tops(s):
    """正規化済み強さ s(6) から各券種の上位3点（組番文字列）と自信を返す。"""
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
    trio = {}
    for (a, b, c), p in tri.items():
        k = tuple(sorted((a, b, c)))
        trio[k] = trio.get(k, 0.0) + p

    def t3(d):
        return ["-".join(map(str, k)) for k, _ in sorted(d.items(), key=lambda x: -x[1])[:3]]
    return ({"3連単": t3(tri), "3連複": t3(trio), "2連単": t3(exa), "2連複": t3(quin)},
            float(s.max() / S))


log("データ準備・学習（2024末まで／節間モーター込み）...")
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
mc = ["date", "venue", "race", "boat", "motor_setsu_top2", "motor_setsu_n"]
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
va = va.copy()
va["pred"] = model.predict(va[FE], num_iteration=model.best_iteration)

log("実払戻(payouts_all.csv)を読み込み ...")
pall = pd.read_csv("data/csv/payouts_all.csv")
pall = pall[pall["date"] >= "2025-01-01"]
WIN = {}   # (date,venue,race,bet_type) -> (combo, payout)
for date, v, r, bt, combo, pay in zip(pall["date"], pall["venue"], pall["race"],
                                      pall["bet_type"], pall["combo"], pall["payout"]):
    if bt in ("3連単", "3連複", "2連単", "2連複"):
        WIN[(date, int(v), int(r), bt)] = (combo, int(pay))

log("各レースで券種別 上位3点を算出し、実払戻と突合 ...")
TYPES = ["3連単", "3連複", "2連単", "2連複"]
agg = {t: {"n": 0, "hit": 0, "ret": 0} for t in TYPES}   # 上位3点・各100円
rec_agg = {"n": 0, "hit": 0, "ret": 0, "cost": 0}
rec_count = {t: 0 for t in TYPES}

for (dt, v, r), g in va.groupby(["date", "venue", "race"]):
    g = g.sort_values("boat")
    if len(g) != 6:
        continue
    raw = np.clip(g["pred"].values, 1e-9, None)
    s = raw / raw.sum()
    tops, conf = bet_tops(s)
    # 券種別：上位3点を各100円
    for t in TYPES:
        w = WIN.get((dt, int(v), int(r), t))
        if w is None:
            continue
        agg[t]["n"] += 1
        if w[0] in tops[t]:
            agg[t]["hit"] += 1
            agg[t]["ret"] += w[1]
    # おすすめ券種（自信で出し分け）
    rt = ("3連単" if conf >= 0.55 else "3連複" if conf >= 0.45
          else "2連単" if conf >= 0.38 else "2連複")
    rec_count[rt] += 1
    w = WIN.get((dt, int(v), int(r), rt))
    if w is not None:
        rec_agg["n"] += 1; rec_agg["cost"] += 300
        if w[0] in tops[rt]:
            rec_agg["hit"] += 1; rec_agg["ret"] += w[1]

log("\n========== A. 券種別 検証結果（2025以降・上位3点を各100円買い）==========")
log(f"{'券種':<6}{'対象R':>8}{'的中率':>9}{'回収率':>9}")
for t in TYPES:
    a = agg[t]
    if a["n"] == 0:
        continue
    hit = a["hit"]/a["n"]*100
    roi = a["ret"]/(a["n"]*300)*100
    log(f"{t:<6}{a['n']:>8,}{hit:>8.1f}%{roi:>8.1f}%")

log("\n--- おすすめ券種ロジック（自信で1券種に出し分け）---")
log("おすすめ内訳: " + " / ".join(f"{t} {rec_count[t]:,}R" for t in TYPES))
if rec_agg["n"]:
    log(f"  的中率 {rec_agg['hit']/rec_agg['n']*100:.1f}%   "
        f"回収率 {rec_agg['ret']/rec_agg['cost']*100:.1f}%   "
        f"(対象 {rec_agg['n']:,}R)")
log("\n※回収率100%超なら理論上プラス。控除率25%のため75%が基準線。")
log("※本番モデルは不変。リーク無し（2024末まで学習→2025以降で検証）。")
