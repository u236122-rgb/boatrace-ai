# -*- coding: utf-8 -*-
"""
core.py  ―  競艇予想エンジン
------------------------------------------------------------
build_model.py が作った model/ を読み込み、
「場・6選手の登録番号・天候・風・波」から

  ・各艇(1～6号艇)の 1着確率 / 6着(最下位)確率
  ・3連単120通りの確率（Plackett-Luce）
  ・本命5点 / 中穴5点 / 大穴5点

を算出する。GUI(予想GUI.py)から呼び出される。
"""
import os
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "model")


def race_advice(race_no, title=""):
    """レース番号・レース名から『狙い目アドバイス』(参考)を返す。予想確率は変えない。
    番組表から番号やレース名が分かる時だけ出す。分からなければ空文字。"""
    t = title or ""
    if "優勝戦" in t and "準" not in t:
        return ("🎯 優勝戦：実力者が集結する一戦。1号艇中心だが、"
                "センター勢の一発（まくり差し）に警戒（荒れ寄り）。")
    if "準優" in t:
        return ("🎯 準優勝戦：勝ち上がり同士で堅・荒れに振れやすい。"
                "センター勢に注意（やや荒れ寄り）。")
    try:
        rno = int(race_no)
    except (TypeError, ValueError):
        return ""
    if rno in (1, 3):
        return ("🎯 1R・3Rはメンバーが堅めになりやすい傾向（企画・予選序盤）。"
                "本命寄りで。")
    return ""


def naruto_wind_advice(winddir, wind):
    """鳴門専用：風向・風速から、検証で裏付けが取れた狙い目を返す。
    風向は絶対方位で代用（北西＝追い風／南東＝向かい風と対応づけ）。
    検証済みのものだけ（4R荒れ説・気温説は不採用）。"""
    try:
        w = float(wind)
    except (TypeError, ValueError):
        return ""
    if winddir == "北西":           # 追い風
        if 4 <= w <= 6:
            return ("🌬 追い風(北西)4〜6m：1コースが約42%に低下。"
                    "2コースの差し・3コースのまくりを警戒。")
        if w >= 7:
            return ("🌬 追い風(北西)7m以上：1コースが約46%に回復。"
                    "むしろイン寄りに戻る傾向。")
    if winddir == "南東":           # 向かい風
        if 4 <= w <= 6:
            return ("🌬 向かい風(南東)4〜6m：センター（3・4コース）の"
                    "1着率が上昇。まくり差しに注意。")
        if w >= 7:
            return ("🌬 向かい風(南東)7m以上：センター勢が大きく台頭。"
                    "3・4コースを厚めに。")
    return ""


class Predictor:
    def __init__(self, model_dir=None):
        d = os.path.join(BASE, model_dir) if model_dir else OUT
        self.model = lgb.Booster(model_file=os.path.join(d, "lgb_model.txt"))
        t = joblib.load(os.path.join(d, "tables.pkl"))
        self.features = t["features"]
        self.grade_map = t["grade_map"]
        self.weather_list = t["weather_list"]
        self.winddir_list = t["winddir_list"]
        self.venue_names = t["venue_names"]
        self.defaults = t["defaults"]
        self.racer_global = t["racer_global"]
        self.venue_course = t["venue_course"]
        self.rc_table = t["rc_table"]        # (racer_id,course)->(win,top2,st)
        self.rv_table = t["rv_table"]        # (racer_id,venue)->(win,top2)
        self.rr_avg_st = t["rr_avg_st"]      # racer_id->avg_st
        self.recent_win = t["recent_win"]    # racer_id->直近20走勝率
        self.glob = t["glob"]
        self.frame_course = t.get("frame_course", {})  # 場->6x6(枠->進入コース確率)
        self.racer_frame_course = t.get("racer_frame_course", {})  # (racer,枠)->6
        self.valid_acc = t.get("valid_acc", 0.0)
        self.data_max_date = t.get("data_max_date", "")
        self.n_races = t.get("n_races", 0)
        self.is_naruto = t.get("is_naruto", False)
        self.naruto_course_rate = t.get("naruto_course_rate", [])
        self._motor_hist = None       # 節間モーター用（results.csvから遅延読込）

    # ---- 節間モーター成績（その節の前日までの2連率）----
    def _ensure_motor_hist(self):
        """results.csv の直近120日分だけ読み、(場,日,モーター)→(2連数,試走数) を作る。
        予想時に『この節の前日まで』のモーター2連率を出すための土台。重いので遅延。"""
        if self._motor_hist is not None:
            return
        path = os.path.join(BASE, "data", "csv", "results.csv")
        try:
            r = pd.read_csv(path, usecols=["date", "venue", "motor", "finish"])
        except Exception:
            self._motor_hist = ({}, {}); return
        r = r.dropna(subset=["finish", "motor"])
        if len(r):
            cut = (pd.to_datetime(r["date"].max()) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
            r = r[r["date"] >= cut]
        r["top2"] = (r["finish"] <= 2).astype(int)
        grp = r.groupby(["venue", "date", "motor"])["top2"].agg(["sum", "size"])
        daily = {(int(v), d, int(m)): (int(s), int(n))
                 for (v, d, m), (s, n) in grp.iterrows()}
        vdates = {}
        for (v, d, _m) in daily:
            vdates.setdefault(v, set()).add(d)
        vdates = {v: sorted(s) for v, s in vdates.items()}
        self._motor_hist = (daily, vdates)

    def motor_setsu(self, venue, motor_no, date_iso):
        """その節（連続開催・3日以上空けば別節）の“前日まで”の
        モーター2連率(0-1)と試走数を返す。取れなければ (None, 0)。"""
        import bisect
        self._ensure_motor_hist()
        daily, vdates = self._motor_hist
        try:
            venue = int(venue); motor_no = int(motor_no)
        except (TypeError, ValueError):
            return (None, 0)
        dts = vdates.get(venue)
        if not dts:
            return (None, 0)
        i = bisect.bisect_left(dts, date_iso)       # dts[i-1] が直前の開催日
        ssum = scnt = 0
        nextd = date_iso
        j = i - 1
        while j >= 0:
            d = dts[j]
            if (pd.to_datetime(nextd) - pd.to_datetime(d)).days > 3:
                break                                # 節が途切れた
            rec = daily.get((venue, d, motor_no))
            if rec:
                ssum += rec[0]; scnt += rec[1]
            nextd = d
            j -= 1
        if scnt == 0:
            return (None, 0)
        return (ssum / scnt, scnt)

    # ---- 進入予想：場の傾向＋選手の前づけ癖から進入コースを推定 ----
    def predict_courses(self, venue, racer_ids=None):
        """場の枠別傾向と、各選手の枠別「前づけの癖」をブレンドし、
        最も起こりやすい進入(コース割り当て)をハンガリアン法で1つ返す。
        courses[i] = i号艇(枠i+1)の予想進入コース。"""
        vmat = self.frame_course.get(int(venue))
        if not vmat:
            return [1, 2, 3, 4, 5, 6]
        exp_course = np.zeros(6)                      # 各艇の「期待進入コース」
        for i in range(6):
            vp = np.array(vmat[i], dtype=float)
            rp = None
            if racer_ids is not None:
                rp = self.racer_frame_course.get((int(racer_ids[i]), i + 1))
            d = 0.2 * vp + 0.8 * np.array(rp, dtype=float) if rp is not None else vp
            s = d.sum()
            d = d / s if s > 0 else vp
            exp_course[i] = float((np.arange(1, 7) * d).sum())
        # 期待進入が内(小さい)の艇から順に 1,2,…コースを割り当て。
        # 同点は枠順を優先（安定化）。前づけ癖のある外枠が内に食い込む。
        order = sorted(range(6), key=lambda i: (exp_course[i], i))
        courses = [0] * 6
        for c, i in enumerate(order):
            courses[i] = c + 1
        return courses

    # ---- 選手情報の取得（登録番号→成績）----
    def racer_info(self, racer_id, venue, course):
        """登録番号・場・コースから、その選手の成績ディクショナリを返す。
        コース別実績・当地実績・平均ST・直近の調子まで自動取得する。"""
        info = dict(self.defaults)
        found = False
        name = f"No.{racer_id}"
        if racer_id in self.racer_global.index:
            g = self.racer_global.loc[racer_id]
            name = g["name"]
            info["grade_num"] = float(g["grade_num"])
            info["age"] = float(g["age"])
            info["weight"] = float(g["weight"])
            info["win_nat"] = float(g["win_nat"])
            info["top2_nat"] = float(g["top2_nat"])
            info["win_loc"] = float(g["win_loc"])
            info["top2_loc"] = float(g["top2_loc"])
            info["motor_top2"] = float(g["motor_top2"])
            info["boat_top2"] = float(g["boat_top2"])
            found = True

        # コース別実績（無ければそのコースの全体平均）
        rc = self.rc_table.get((racer_id, course))
        info["rc_win"] = rc[0] if rc else self.glob["course_win"].get(course, self.glob["g_win"])
        info["rc_top2"] = rc[1] if rc else self.glob["course_top2"].get(course, self.glob["g_top2"])
        info["rc_st"] = rc[2] if rc else self.glob["course_st"].get(course, self.glob["g_st"])
        # 当地実績（無ければ全国系で代用）
        rv = self.rv_table.get((racer_id, venue))
        info["rv_win"] = rv[0] if rv else self.glob["g_win"]
        info["rv_top2"] = rv[1] if rv else self.glob["g_top2"]
        info["_rv_found"] = rv is not None
        info["_rc_found"] = rc is not None
        # 平均ST・直近の調子
        info["r_avg_st"] = self.rr_avg_st.get(racer_id, self.glob["g_st"])
        info["r_recent_win"] = self.recent_win.get(racer_id, self.glob["g_win"])

        info["_name"] = name
        info["_found"] = found
        return info

    # ---- メイン予想 ----
    def predict(self, venue, racer_ids, weather, wind_dir, wind, wave,
                courses=None, exhibit=None, motor_nos=None, date=None):
        """
        venue      : 場コード(int 1-24)
        racer_ids  : [1号艇, ..., 6号艇] の登録番号(int) 6個
        weather    : 天候(str)  例 "晴"
        wind_dir   : 風向(str)  例 "北"
        wind       : 風速(float, m)
        wave       : 波高(float, cm)
        courses    : 進入コース[6]（i号艇の実際の進入）。None=枠なり[1..6]
        exhibit    : 展示タイム[6]（任意・直前情報）。None=未使用。
        戻り値 dict
        """
        weather_code = self.weather_list.index(weather) if weather in self.weather_list else 1
        winddir_code = self.winddir_list.index(wind_dir) if wind_dir in self.winddir_list else 0
        if courses is None:
            courses = [i + 1 for i in range(6)]   # 枠なり

        # 展示タイム（手入力・任意）。無い艇は中央値で埋める＝中立。
        ex_med = float(self.defaults.get("exhibit_f", 6.8))
        exf = [ex_med] * 6
        ex_used = False
        if exhibit is not None:
            for i, e in enumerate(exhibit[:6]):
                try:
                    exf[i] = float(e)
                    ex_used = True
                except (TypeError, ValueError):
                    pass

        rows = []
        infos = []
        for i, rid in enumerate(racer_ids):
            boat = i + 1                  # 枠番（既知・固定）
            course = int(courses[i])      # 実際の進入コース（進入予想で変わる）
            info = self.racer_info(int(rid), int(venue), course)
            info["_course"] = course
            # 節間モーター成績：モーター番号と日付が分かる時だけ実値、無ければ節初日扱い
            ms_top2 = float(info["motor_top2"]) / 100.0   # 既定＝番組表の値(比率)
            ms_n = 0.0
            if motor_nos is not None and date is not None and motor_nos[i] is not None:
                v, n = self.motor_setsu(int(venue), motor_nos[i], date)
                if v is not None:
                    ms_top2, ms_n = v, float(n)
            info["_motor_setsu_top2"] = ms_top2
            info["_motor_setsu_n"] = ms_n
            infos.append(info)
            rows.append({
                "boat": boat, "venue": int(venue),
                "grade_num": info["grade_num"], "age": info["age"],
                "weight": info["weight"],
                "win_nat": info["win_nat"], "top2_nat": info["top2_nat"],
                "win_loc": info["win_loc"], "top2_loc": info["top2_loc"],
                "motor_top2": info["motor_top2"], "boat_top2": info["boat_top2"],
                "weather_code": weather_code, "winddir_code": winddir_code,
                "wind": float(wind), "wave": float(wave),
                "rc_win": info["rc_win"], "rc_top2": info["rc_top2"],
                "rc_st": info["rc_st"], "rv_win": info["rv_win"],
                "rv_top2": info["rv_top2"], "r_avg_st": info["r_avg_st"],
                "exhibit_f": exf[i],
                "motor_setsu_top2": ms_top2, "motor_setsu_n": ms_n,
            })
        X = pd.DataFrame(rows)
        # レース内相対特徴
        X["win_nat_rel"] = X["win_nat"] - X["win_nat"].mean()
        X["top2_nat_rel"] = X["top2_nat"] - X["top2_nat"].mean()
        X["motor_rel"] = X["motor_top2"] - X["motor_top2"].mean()
        X["rc_win_rel"] = X["rc_win"] - X["rc_win"].mean()
        # 展示タイムのレース内相対（速いほど小さい→マイナス）。無入力なら全て0。
        X["exhibit_rel"] = X["exhibit_f"] - X["exhibit_f"].mean()

        raw = self.model.predict(X[self.features])
        raw = np.clip(raw, 1e-6, None)

        # 1着確率（6艇で正規化）= Plackett-Luce の強さ
        strength = raw / raw.sum()

        # --- 各艇の1着・6着(最下位)確率を全順列から算出 ---
        idx = list(range(6))
        p_first = np.zeros(6)
        p_last = np.zeros(6)
        # 3連単(上位3着)の確率も同時に集計
        trifecta = {}
        s = strength.copy()
        for perm in itertools.permutations(idx):
            # Plackett-Luce: 残り集合から順に選ぶ
            p = 1.0
            remaining = s.copy()
            total = remaining.sum()
            for pos, b in enumerate(perm):
                p *= remaining[b] / total
                total -= remaining[b]
                remaining[b] = 0.0
            p_first[perm[0]] += p   # （後で正規化されるが）厳密には perm[0] のみで決まる
            p_last[perm[5]] += p
            key = (perm[0] + 1, perm[1] + 1, perm[2] + 1)
            trifecta[key] = trifecta.get(key, 0.0) + p

        # p_first は strength と一致するはず（正規化）
        p_first = strength.copy()
        p_last = p_last / p_last.sum()

        # ---- 3連単を確率順に整列し、想定配当を付与 ----
        combos = []
        for key, prob in trifecta.items():
            prob = max(prob, 1e-9)
            # 想定配当(円)：控除率25%を反映した概算オッズ
            est_payout = int(round(0.75 / prob * 100))
            combos.append({
                "combo": "%d-%d-%d" % key,
                "prob": prob,
                "payout": est_payout,
            })
        combos.sort(key=lambda x: -x["prob"])

        honmei, chuuana, ooana = self._classify(combos)

        # ---- 他券種（3連複/2連単/2連複）を強さから算出 ----
        st = strength
        exacta = {}                       # 2連単 a→b
        for a in range(6):
            den = 1.0 - st[a]
            for b in range(6):
                if b == a:
                    continue
                exacta[(a + 1, b + 1)] = st[a] * st[b] / den if den > 1e-9 else 0.0
        quinella = {}                     # 2連複 {a,b}
        for a in range(6):
            for b in range(a + 1, 6):
                quinella[(a + 1, b + 1)] = exacta[(a + 1, b + 1)] + exacta[(b + 1, a + 1)]
        trio = {}                         # 3連複 {a,b,c}
        for key, pr in trifecta.items():
            t = tuple(sorted(key))
            trio[t] = trio.get(t, 0.0) + pr

        def topk(d, k=3):
            out = []
            for combo, pr in sorted(d.items(), key=lambda x: -x[1])[:k]:
                pr = max(pr, 1e-9)
                out.append({"combo": "-".join(map(str, combo)), "prob": pr,
                            "payout": int(round(0.75 / pr * 100)),
                            "be_odds": round(1.0 / pr, 1)})   # 買い基準オッズ(1/確率)
            return out

        bet_types = {"3連単": topk(trifecta), "3連複": topk(trio),
                     "2連単": topk(exacta), "2連複": topk(quinella)}

        # ---- おすすめ券種（自信＝最有力艇の1着確率で判定）----
        conf = float(strength.max())
        if conf >= 0.55:
            rec = ("3連単", "自信あり：本命がはっきり。的中なら高配当の3連単で勝負。")
        elif conf >= 0.45:
            rec = ("3連複", "やや自信：着順までは不安。3着内を手広く取る3連複が手堅い。")
        elif conf >= 0.38:
            rec = ("2連単", "自信は中程度：上位2艇の順番に絞る2連単が無難。")
        else:
            rec = ("2連複", "波乱含み：着順は読みにくい。まず当てにいくなら2連複。")
        # 各券種のトップ的中率も付けておく
        recommend = {"type": rec[0], "reason": rec[1], "conf": conf,
                     "top_prob": {k: (v[0]["prob"] if v else 0) for k, v in bet_types.items()}}

        # ---- 戦闘力スコア ----
        # 総合戦闘力：コースに関係ない選手の地力（全国勝率＋全国2連率）を0-100に。
        # レース別戦闘力：このレース・この枠での1着力（=strength）を%と相対スコアで。
        smax = float(strength.max()) if strength.max() > 0 else 1.0
        power = []
        for i, info in enumerate(infos):
            wn = float(info.get("win_nat", 5.0))
            t2 = float(info.get("top2_nat", 33.0))
            nw = min(max((wn - 3.0) / 5.5, 0.0), 1.0)   # 全国勝率 3.0→0, 8.5→100
            nt = min(max(t2 / 65.0, 0.0), 1.0)          # 全国2連率 65%→100
            overall = int(round((0.6 * nw + 0.4 * nt) * 100))
            pi = float(strength[i])
            power.append({
                "boat": i + 1,
                "overall": overall,                                  # 総合(0-100)
                "race_pct": round(pi * 100, 1),                      # このレース1着力%
                "race_score": int(round(pi / smax * 100)),           # レース内相対(最強=100)
                "tansho_be": round(1.0 / pi, 2) if pi > 1e-9 else 999.0,  # 単勝・買い基準オッズ
            })

        return {
            "strength": strength,         # =1着確率
            "power": power,               # 戦闘力（総合＋レース別）
            "p_first": p_first,
            "p_last": p_last,             # 6着(最下位)になりやすさ
            "infos": infos,
            "combos": combos,
            "honmei": honmei,
            "chuuana": chuuana,
            "ooana": ooana,
            "bet_types": bet_types,       # 券種別トップ3点
            "recommend": recommend,       # おすすめ券種
            "venue_name": self.venue_names.get(int(venue), str(venue)),
            "courses": list(courses),     # 採用した進入コース
            "exhibit_used": ex_used,      # 展示補正をかけたか
        }

    # ---- 当日の全レースを一括予想 ----
    def predict_program(self, races, weather, wind_dir, wind, wave, progress=None,
                        date_iso=None):
        """番組表(program.parse_program の出力)を受け取り、各レースを予想して
        サマリ一覧を返す。progress(done, total) で進捗通知。
        date_iso(YYYY-MM-DD)があれば節間モーター成績も反映する。"""
        out = []
        n = len(races)
        for i, race in enumerate(races):
            rs = sorted(race["racers"], key=lambda b: b["boat"])
            ids = [x["racer_id"] for x in rs]
            motor_nos = [x.get("motor_no") for x in rs]
            r = self.predict(race["venue"], ids, weather, wind_dir, wind, wave,
                             motor_nos=motor_nos, date=date_iso)
            win = r["strength"]
            top = float(win.max())
            best = int(win.argmax()) + 1
            worst = int(r["p_last"].argmax()) + 1
            if top >= 0.55:
                kind = "堅い"
            elif top >= 0.40:
                kind = "やや混戦"
            else:
                kind = "波乱(大穴注意)"
            out.append({
                "venue": race["venue"], "venue_name": r["venue_name"],
                "race": race["race"], "title": race.get("title", ""),
                "honmei": r["honmei"][0],
                "ooana": r["ooana"][0] if r["ooana"] else None,
                "best": best, "worst": worst, "top": top, "kind": kind,
                "ids": ids, "full": r,
            })
            if progress:
                progress(i + 1, n)
        return out

    # ---- 本命/中穴/大穴の振り分け（重複なし・想定配当帯で分類）----
    @staticmethod
    def _classify(combos):
        """combos は確率降順。3グループは買い目が重複しないように分ける。
        本命 = 最も当たりやすい5点。
        中穴 = 本命以外で想定配当 中位(1500〜6000円)の確率上位5点。
        大穴 = 残りで高配当(6000円〜)の確率上位5点（一発逆転）。"""
        used = set()

        def take(pred, n=5):
            picked = []
            for c in combos:
                if c["combo"] in used:
                    continue
                if pred(c):
                    picked.append(c)
                    used.add(c["combo"])
                    if len(picked) >= n:
                        break
            return picked

        honmei = take(lambda c: True)                                   # 確率上位5
        chuuana = take(lambda c: 1500 <= c["payout"] < 6000)           # 中配当
        if len(chuuana) < 5:                                            # 不足は確率順で補完
            chuuana += take(lambda c: True, 5 - len(chuuana))
        ooana = take(lambda c: c["payout"] >= 6000)                    # 高配当
        if len(ooana) < 5:
            ooana += take(lambda c: True, 5 - len(ooana))
        return honmei, chuuana, ooana


if __name__ == "__main__":
    # 簡易動作確認
    p = Predictor()
    print("データ最終日:", p.data_max_date, "/ 本命的中率:", round(p.valid_acc*100, 1), "%")
    # びわこ(11)で適当な登録番号6名・条件で予想
    sample = list(p.racer_global.index[:6])
    r = p.predict(11, sample, "晴", "北", 3, 5)
    print("場:", r["venue_name"])
    for i in range(6):
        print(f" {i+1}号艇 {r['infos'][i]['_name']:<8} "
              f"1着{r['strength'][i]*100:5.1f}% / 最下位{r['p_last'][i]*100:5.1f}%")
    print("本命:", [c["combo"] for c in r["honmei"]])
    print("中穴:", [c["combo"] for c in r["chuuana"]])
    print("大穴:", [c["combo"] for c in r["ooana"]])
