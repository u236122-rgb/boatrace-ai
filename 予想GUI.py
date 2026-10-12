# -*- coding: utf-8 -*-
"""
予想GUI.py  ―  競艇AI予想ツール（GUI）
============================================================
場・6選手(登録番号)・天候・風・波を選んで「予想する」を押すと、

  ・各艇の【1着になりやすさ】【最下位(6着)になりやすさ】
  ・3連単 本命5点 / 中穴5点 / 大穴5点（想定配当つき）
  ・大穴が狙えるレースかどうかの判定コメント

を表示する。琵琶湖のAIに負けない本格データ予想。

  起動:  python 予想GUI.py
  （事前に build_model.py を実行してモデルを作っておくこと）
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

try:
    from core import Predictor, race_advice, naruto_wind_advice
    import program
except Exception as e:
    tk.Tk().withdraw()
    messagebox.showerror("起動エラー",
                         "モデルの読み込みに失敗しました。\n"
                         "先に  python build_model.py  を実行してください。\n\n"
                         f"詳細: {e}")
    sys.exit(1)

# ---- 配色 ----
BG = "#0f1626"
PANEL = "#1b2540"
ACCENT = "#36c2ff"
HONMEI = "#ffd43b"   # 本命=金
CHUUANA = "#69db7c"  # 中穴=緑
OOANA = "#ff6b6b"    # 大穴=赤
TEXT = "#e9eefb"
SUB = "#9fb0d0"
FONT = "Yu Gothic UI"


class App(tk.Tk):
    def __init__(self, model_dir=None, naruto=False):
        super().__init__()
        self.naruto = naruto
        self.pred = Predictor(model_dir)
        self._race_no = None       # 狙い目アドバイス用のレース文脈
        self._race_title = ""
        self.title("🌊 鳴門特化AI予想" if naruto else "競艇AI予想  〜 打倒・琵琶湖AI 〜")
        self.configure(bg=BG)
        self.geometry("1080x760")
        self.minsize(980, 680)

        self._build_style()
        self._build_header()
        self._build_inputs()
        self._build_output()

    # ---------------------------------------------------- スタイル
    def _build_style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        # 入力欄は「明るい背景＋濃い文字」で見やすく
        FIELD = "#eaf2ff"     # 明るい水色がかった白
        FIELD_TX = "#10203a"  # 濃紺の文字
        self.FIELD, self.FIELD_TX = FIELD, FIELD_TX
        st.configure("TCombobox", fieldbackground=FIELD, background=FIELD,
                     foreground=FIELD_TX, arrowcolor="#1b66b3", padding=5,
                     bordercolor="#1b66b3", relief="flat")
        # readonly 状態でも背景・文字色を固定（反転で見えなくなるのを防ぐ）
        st.map("TCombobox",
               fieldbackground=[("readonly", FIELD), ("disabled", "#33405e")],
               foreground=[("readonly", FIELD_TX), ("disabled", SUB)],
               selectbackground=[("readonly", FIELD)],
               selectforeground=[("readonly", FIELD_TX)],
               background=[("readonly", FIELD)],
               arrowcolor=[("readonly", "#1b66b3")])
        # ドロップダウン一覧も明るく
        self.option_add("*TCombobox*Listbox.background", FIELD)
        self.option_add("*TCombobox*Listbox.foreground", FIELD_TX)
        self.option_add("*TCombobox*Listbox.selectBackground", "#1b66b3")
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.option_add("*TCombobox*Listbox.font", (FONT, 11))

    def _lbl(self, parent, text, **kw):
        return tk.Label(parent, text=text, bg=kw.pop("bg", PANEL),
                        fg=kw.pop("fg", TEXT), font=(FONT, kw.pop("fs", 10)), **kw)

    # ---------------------------------------------------- ヘッダー
    def _build_header(self):
        h = tk.Frame(self, bg=BG)
        h.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(h, text="🚤 競艇AI予想ツール", bg=BG, fg=ACCENT,
                 font=(FONT, 20, "bold")).pack(side="left")
        info = (f"学習データ {self.pred.n_races:,}レース"
                f"（最終 {self.pred.data_max_date}）  /  "
                f"本命1着的中率 {self.pred.valid_acc*100:.1f}%")
        tk.Label(h, text=info, bg=BG, fg=SUB,
                 font=(FONT, 10)).pack(side="right", pady=8)

    # ---------------------------------------------------- 入力
    def _build_inputs(self):
        p = tk.Frame(self, bg=PANEL, bd=0)
        p.pack(fill="x", padx=16, pady=6)

        # ===== かんたん入力：番組表からR指定でワンクリック自動入力 =====
        load = tk.Frame(p, bg="#24314f")
        load.pack(fill="x", padx=14, pady=(12, 2))
        tk.Label(load, text="📥 かんたん入力：", bg="#24314f", fg=HONMEI,
                 font=(FONT, 11, "bold")).pack(side="left", padx=(8, 6), pady=6)
        try:
            self._prog_dates = program.list_program_dates()
        except Exception:
            self._prog_dates = []
        tk.Label(load, text="日付", bg="#24314f", fg=SUB, font=(FONT, 10)).pack(side="left")
        self.load_date = ttk.Combobox(load, state="readonly", width=12, font=(FONT, 10),
                                      values=[program.yymmdd_to_iso(d) for d in self._prog_dates])
        if self._prog_dates:
            self.load_date.current(0)
        self.load_date.pack(side="left", padx=(2, 10))
        tk.Label(load, text="R", bg="#24314f", fg=SUB, font=(FONT, 10)).pack(side="left")
        self.load_race = ttk.Combobox(load, state="readonly", width=5, font=(FONT, 10),
                                      values=[f"{i}R" for i in range(1, 13)])
        self.load_race.current(0)
        self.load_race.pack(side="left", padx=(2, 10))
        tk.Button(load, text="この場・R を読み込む", command=self.load_from_program,
                  bg=ACCENT, fg="#06223a", font=(FONT, 10, "bold"), relief="flat",
                  padx=12, pady=3, cursor="hand2").pack(side="left", padx=4)
        tk.Label(load, text="← 上の「場」を選んでから押すと6名を自動入力",
                 bg="#24314f", fg=SUB, font=(FONT, 9)).pack(side="left", padx=6)

        inner = tk.Frame(p, bg=PANEL)
        inner.pack(fill="x", padx=14, pady=12)

        # 場
        self._lbl(inner, "場（レース場）", fs=10, fg=SUB).grid(row=0, column=0, sticky="w")
        names = self.pred.venue_names
        self.venue_box = ttk.Combobox(
            inner, state="readonly", width=14, font=(FONT, 11),
            values=[f"{k:2d}: {v}" for k, v in names.items()])
        # 鳴門特化版は場を鳴門(14)に固定、通常版はびわこ(11)
        if self.naruto:
            vkeys = list(names.keys())
            self.venue_box.current(vkeys.index(14))
            self.venue_box.configure(state="disabled")
        else:
            self.venue_box.current(10)
        self.venue_box.grid(row=1, column=0, sticky="w", padx=(0, 24))

        # 天候・風向・風速・波
        self._lbl(inner, "天候", fs=10, fg=SUB).grid(row=0, column=1, sticky="w")
        self.weather_box = ttk.Combobox(inner, state="readonly", width=7,
                                        font=(FONT, 11), values=self.pred.weather_list)
        self.weather_box.current(0)
        self.weather_box.grid(row=1, column=1, sticky="w", padx=(0, 14))

        self._lbl(inner, "風向", fs=10, fg=SUB).grid(row=0, column=2, sticky="w")
        self.winddir_box = ttk.Combobox(inner, state="readonly", width=7,
                                        font=(FONT, 11), values=self.pred.winddir_list)
        self.winddir_box.current(0)
        self.winddir_box.grid(row=1, column=2, sticky="w", padx=(0, 14))

        self._lbl(inner, "風速(m)", fs=10, fg=SUB).grid(row=0, column=3, sticky="w")
        self.wind_var = tk.StringVar(value="3")
        tk.Spinbox(inner, from_=0, to=18, textvariable=self.wind_var, width=5,
                   font=(FONT, 11), bg=self.FIELD, fg=self.FIELD_TX, justify="center",
                   buttonbackground="#cdddf5", relief="flat",
                   insertbackground=self.FIELD_TX).grid(row=1, column=3, sticky="w", padx=(0, 14))

        self._lbl(inner, "波高(cm)", fs=10, fg=SUB).grid(row=0, column=4, sticky="w")
        self.wave_var = tk.StringVar(value="3")
        tk.Spinbox(inner, from_=0, to=40, textvariable=self.wave_var, width=5,
                   font=(FONT, 11), bg=self.FIELD, fg=self.FIELD_TX, justify="center",
                   buttonbackground="#cdddf5", relief="flat",
                   insertbackground=self.FIELD_TX).grid(row=1, column=4, sticky="w")

        # 選手（登録番号）入力 6艇
        racer = tk.Frame(p, bg=PANEL)
        racer.pack(fill="x", padx=14, pady=(4, 12))
        self._lbl(racer, "選手の登録番号を入力（名前ではなく番号）", fs=10, fg=SUB)\
            .grid(row=0, column=0, columnspan=12, sticky="w", pady=(0, 4))
        self.racer_vars = []
        self.name_labels = []
        colors = ["#ffffff", "#1e1e1e", "#ff3b3b", "#3b6bff",
                  "#ffd400", "#33cc44"]  # 競艇の艇カラー
        for i in range(6):
            box = tk.Frame(racer, bg=colors[i], padx=2, pady=2)
            box.grid(row=1, column=i, padx=5, sticky="w")
            inn = tk.Frame(box, bg=PANEL)
            inn.pack()
            tk.Label(inn, text=f"{i+1}号艇", bg=PANEL, fg=colors[i] if i not in (1,) else "#cccccc",
                     font=(FONT, 10, "bold")).pack()
            v = tk.StringVar()
            e = tk.Entry(inn, textvariable=v, width=8, justify="center",
                         font=(FONT, 13), bg="#26345c", fg=TEXT, insertbackground=TEXT)
            e.pack(pady=2)
            v.trace_add("write", lambda *a, idx=i: self._update_name(idx))
            nl = tk.Label(inn, text="（未入力）", bg=PANEL, fg=SUB, font=(FONT, 9), width=10)
            nl.pack()
            self.racer_vars.append(v)
            self.name_labels.append(nl)

        # ボタン
        btns = tk.Frame(p, bg=PANEL)
        btns.pack(fill="x", padx=14, pady=(0, 12))
        tk.Button(btns, text="🔮 予想する", command=self.on_predict,
                  bg=ACCENT, fg="#06223a", font=(FONT, 13, "bold"),
                  relief="flat", padx=24, pady=8, cursor="hand2").pack(side="left")
        tk.Button(btns, text="📋 当日まとめ予想", command=self.open_batch_window,
                  bg=HONMEI, fg="#3a2e00", font=(FONT, 12, "bold"),
                  relief="flat", padx=16, pady=8, cursor="hand2").pack(side="left", padx=8)
        tk.Button(btns, text="サンプル選手を入れる", command=self.fill_sample,
                  bg="#2a3a63", fg=TEXT, font=(FONT, 10),
                  relief="flat", padx=12, pady=8, cursor="hand2").pack(side="left", padx=8)
        tk.Button(btns, text="クリア", command=self.clear,
                  bg="#2a3a63", fg=TEXT, font=(FONT, 10),
                  relief="flat", padx=12, pady=8, cursor="hand2").pack(side="left")

        # 進入予想チェック
        self.shinnyuu_var = tk.BooleanVar(value=False)
        tk.Checkbutton(btns, text="進入予想を反映（前づけを考慮）",
                       variable=self.shinnyuu_var, command=self.on_predict,
                       bg=PANEL, fg=TEXT, selectcolor="#24314f",
                       activebackground=PANEL, activeforeground=TEXT,
                       font=(FONT, 10)).pack(side="left", padx=(16, 0))

        # 直前情報（任意）の開閉ボタン
        self.exhibit_open = False
        self.exhibit_btn = tk.Button(btns, text="▸ 直前情報（任意）",
                                     command=self.toggle_exhibit, bg="#2a3a63", fg=SUB,
                                     font=(FONT, 10), relief="flat", padx=10, pady=8,
                                     cursor="hand2")
        self.exhibit_btn.pack(side="right")

        # 直前情報パネル（展示タイム）。既定は非表示・未使用。
        self.exhibit_panel = tk.Frame(p, bg="#101a30")
        self.exhibit_vars = []
        tk.Label(self.exhibit_panel,
                 text="展示タイム（秒）を入れると予想に反映します（速い艇を評価）。"
                      "空欄なら使いません（＝直前情報なし）",
                 bg="#101a30", fg=SUB, font=(FONT, 9)).grid(
                     row=0, column=0, columnspan=12, sticky="w", padx=8, pady=(6, 2))
        for i in range(6):
            tk.Label(self.exhibit_panel, text=f"{i+1}号艇", bg="#101a30", fg=SUB,
                     font=(FONT, 9)).grid(row=1, column=i*2, padx=(8, 2), pady=(0, 8))
            ev = tk.StringVar()
            tk.Entry(self.exhibit_panel, textvariable=ev, width=6, justify="center",
                     font=(FONT, 11), bg=self.FIELD, fg=self.FIELD_TX,
                     insertbackground=self.FIELD_TX).grid(
                         row=1, column=i*2+1, padx=(0, 10), pady=(0, 8))
            self.exhibit_vars.append(ev)

    def _update_name(self, idx):
        s = self.racer_vars[idx].get().strip()
        if not s.isdigit():
            self.name_labels[idx].config(text="（未入力）", fg=SUB)
            return
        rid = int(s)
        venue = self._venue()
        if rid in self.pred.racer_global.index:
            nm = self.pred.racer_global.loc[rid, "name"]
            self.name_labels[idx].config(text=str(nm), fg=TEXT)
        else:
            self.name_labels[idx].config(text="該当なし", fg=OOANA)

    # ---------------------------------------------------- 出力エリア
    def _build_output(self):
        # 進入予想などの情報バー
        self.info_bar = tk.Label(self, text="", bg="#101a30", fg=SUB,
                                 font=(FONT, 10, "bold"), anchor="w", padx=12, pady=6)
        self.info_bar.pack(fill="x", padx=16, pady=(2, 0))

        out = tk.Frame(self, bg=BG)
        out.pack(fill="both", expand=True, padx=16, pady=(2, 14))

        # 左：各艇の確率（スクロール可）
        left = tk.Frame(out, bg=PANEL)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(left, text="各艇の評価", bg=PANEL, fg=ACCENT,
                 font=(FONT, 13, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.boat_area = self._scroll_area(left)

        # 右：本命/中穴/大穴（スクロール可）
        right = tk.Frame(out, bg=PANEL)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        tk.Label(right, text="3連単 おすすめ買い目", bg=PANEL, fg=ACCENT,
                 font=(FONT, 13, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        if self.naruto and self.pred.naruto_course_rate:
            cr = self.pred.naruto_course_rate
            txt = "鳴門 コース別1着率（実データ）： " + " ".join(
                f"{c+1}={cr[c]*100:.1f}%" for c in range(6))
            tk.Label(right, text=txt, bg="#101a30", fg="#9fe0ff",
                     font=(FONT, 9), anchor="w").pack(fill="x", padx=12, pady=(0, 4))
        rin = self._scroll_area(right)            # スクロールの中身フレーム
        self.tip_area = tk.Frame(rin, bg=PANEL)
        self.tip_area.pack(fill="x", padx=2, pady=(0, 6))
        self.comment = tk.Label(rin, text="", bg=PANEL, fg=OOANA,
                                font=(FONT, 11, "bold"), wraplength=440, justify="left")
        self.comment.pack(anchor="w", padx=2, pady=(0, 6))
        disc = ("⚠ 鳴門のバックテスト回収率は最大93〜98%（100%未満）。"
                "舟券は余裕資金の範囲で。" if self.naruto else
                "⚠ 過去検証での回収率は最大93%（100%未満）。"
                "舟券は余裕資金の範囲で。")
        tk.Label(rin, text=disc, bg=PANEL, fg="#d8b24a", font=(FONT, 9),
                 wraplength=440, justify="left").pack(anchor="w", padx=2, pady=(0, 10))

    def _scroll_area(self, parent, bg=PANEL):
        """parent内にスクロール可能な領域を作り、中身を入れるFrameを返す。"""
        canvas = tk.Canvas(parent, bg=bg, highlightthickness=0)
        sb = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=bg)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 8))
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))

        def wheel(e):
            canvas.yview_scroll(int(-e.delta / 120), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        return inner

    # ---------------------------------------------------- 補助
    def _venue(self):
        return int(self.venue_box.get().split(":")[0])

    def fill_sample(self):
        ids = list(self.pred.racer_global.index[:6])
        for v, rid in zip(self.racer_vars, ids):
            v.set(str(int(rid)))
        self._race_no = None       # 番組由来でない→アドバイスは出さない
        self._race_title = ""

    def clear(self):
        for v in self.racer_vars:
            v.set("")
        for v in getattr(self, "exhibit_vars", []):
            v.set("")
        self._race_no = None
        self._race_title = ""

    def toggle_exhibit(self):
        self.exhibit_open = not self.exhibit_open
        if self.exhibit_open:
            self.exhibit_panel.pack(fill="x", padx=14, pady=(0, 10))
            self.exhibit_btn.config(text="▾ 直前情報（任意）", fg=TEXT)
        else:
            self.exhibit_panel.pack_forget()
            self.exhibit_btn.config(text="▸ 直前情報（任意）", fg=SUB)

    def load_from_program(self):
        """選択中の日付・場・R から、その6名の登録番号を自動入力する。"""
        if not self._prog_dates:
            messagebox.showinfo("かんたん入力", "番組表(data/raw/B)がありません。")
            return
        iso = self.load_date.get()
        yymmdd = iso[2:].replace("-", "")
        rno = int(self.load_race.get().replace("R", ""))
        venue = self._venue()
        try:
            races = program.parse_program(yymmdd)
        except Exception as e:
            messagebox.showerror("エラー", str(e))
            return
        hit = [r for r in races if r["venue"] == venue and r["race"] == rno]
        if not hit:
            vn = self.pred.venue_names.get(venue, venue)
            messagebox.showinfo("かんたん入力",
                                f"{iso} の {vn} {rno}R は番組にありません。\n"
                                "（その日にその場が開催かを確認してください）")
            return
        ids = [x["racer_id"] for x in sorted(hit[0]["racers"], key=lambda b: b["boat"])]
        for v, rid in zip(self.racer_vars, ids):
            v.set(str(int(rid)))
        for v in self.exhibit_vars:
            v.set("")
        # 狙い目アドバイス用に、このレースの番号とレース名を覚えておく
        self._race_no = rno
        self._race_title = hit[0].get("title", "")
        self.on_predict()

    def _exhibit_list(self):
        vals = [v.get().strip() for v in getattr(self, "exhibit_vars", [])]
        if any(s for s in vals):
            return [s if s else "" for s in vals]
        return None

    # ---------------------------------------------------- 当日まとめ予想
    def open_batch_window(self):
        try:
            dates = program.list_program_dates()
        except Exception as e:
            messagebox.showerror("エラー", f"番組表が見つかりません。\n{e}")
            return
        if not dates:
            messagebox.showerror("エラー",
                                 "data/raw/B に番組表(bYYMMDD.txt)がありません。")
            return

        win = tk.Toplevel(self)
        win.title("当日まとめ予想（場を指定）")
        win.configure(bg=BG)
        win.geometry("1040x720")

        # ===== ① 日付・場を選ぶ =====
        bar = tk.Frame(win, bg=PANEL)
        bar.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(bar, text="① 日付", bg=PANEL, fg=ACCENT, font=(FONT, 10, "bold")
                 ).pack(side="left", padx=(8, 2))
        date_box = ttk.Combobox(bar, state="readonly", width=12, font=(FONT, 10),
                                values=[program.yymmdd_to_iso(d) for d in dates])
        date_box.current(0)
        date_box.pack(side="left", padx=(0, 14))
        tk.Label(bar, text="② レース場", bg=PANEL, fg=ACCENT, font=(FONT, 10, "bold")
                 ).pack(side="left", padx=(4, 2))
        venue_box = ttk.Combobox(bar, state="readonly", width=12, font=(FONT, 11))
        venue_box.pack(side="left", padx=(0, 10))

        # ===== ② 条件をあとから入力 =====
        cond = tk.Frame(win, bg=PANEL)
        cond.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(cond, text="③ 条件を入力", bg=PANEL, fg=ACCENT, font=(FONT, 10, "bold")
                 ).pack(side="left", padx=(8, 8))
        tk.Label(cond, text="天候", bg=PANEL, fg=SUB, font=(FONT, 10)).pack(side="left")
        w_box = ttk.Combobox(cond, state="readonly", width=5, font=(FONT, 10),
                             values=self.pred.weather_list)
        w_box.set(self.weather_box.get())
        w_box.pack(side="left", padx=(2, 10))
        tk.Label(cond, text="風向", bg=PANEL, fg=SUB, font=(FONT, 10)).pack(side="left")
        d_box = ttk.Combobox(cond, state="readonly", width=5, font=(FONT, 10),
                             values=self.pred.winddir_list)
        d_box.set(self.winddir_box.get())
        d_box.pack(side="left", padx=(2, 10))
        tk.Label(cond, text="風速(m)", bg=PANEL, fg=SUB, font=(FONT, 10)).pack(side="left")
        wind_e = tk.Spinbox(cond, from_=0, to=18, width=4, font=(FONT, 10),
                            justify="center", bg="#26345c", fg=TEXT, buttonbackground=PANEL)
        wind_e.delete(0, "end"); wind_e.insert(0, self.wind_var.get())
        wind_e.pack(side="left", padx=(2, 10))
        tk.Label(cond, text="波高(cm)", bg=PANEL, fg=SUB, font=(FONT, 10)).pack(side="left")
        wave_e = tk.Spinbox(cond, from_=0, to=40, width=4, font=(FONT, 10),
                            justify="center", bg="#26345c", fg=TEXT, buttonbackground=PANEL)
        wave_e.delete(0, "end"); wave_e.insert(0, self.wave_var.get())
        wave_e.pack(side="left", padx=(2, 14))
        tk.Button(cond, text="④ 予想開始 ▶", command=lambda: run(),
                  bg=ACCENT, fg="#06223a", font=(FONT, 12, "bold"),
                  relief="flat", padx=20, pady=4, cursor="hand2").pack(side="left")
        tk.Button(cond, text="選択Rに条件適用＆再予想", command=lambda: apply_one(),
                  bg=CHUUANA, fg="#06311a", font=(FONT, 10, "bold"), relief="flat",
                  padx=12, pady=4, cursor="hand2").pack(side="left", padx=8)
        tk.Button(cond, text="CSV保存", command=lambda: save_csv(), bg="#2a3a63",
                  fg=TEXT, font=(FONT, 10), relief="flat", padx=10, pady=4,
                  cursor="hand2").pack(side="left", padx=8)

        status = tk.Label(win, text="日付と場を選び、条件を入れて「予想開始」を押してください。",
                          bg=BG, fg=SUB, font=(FONT, 10))
        status.pack(anchor="w", padx=14, pady=(2, 2))

        # ===== 結果（レースごとに本命5/中穴5/大穴5を展開表示）=====
        style = ttk.Style(win)
        style.configure("B.Treeview", background="#16213d", fieldbackground="#16213d",
                        foreground=TEXT, rowheight=24, font=(FONT, 10))
        style.configure("B.Treeview.Heading", background=PANEL, foreground=ACCENT,
                        font=(FONT, 10, "bold"))
        cols = ("買い目", "的中率", "想定配当", "メモ")
        tree = ttk.Treeview(win, columns=cols, show="tree headings", style="B.Treeview")
        tree.heading("#0", text="レース / 種別")
        tree.column("#0", width=330, anchor="w")
        for c, w in zip(cols, (130, 90, 130, 220)):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor="center")
        tree.column("メモ", anchor="w")
        tree.tag_configure("race", background="#243154", font=(FONT, 11, "bold"))
        tree.tag_configure("race_ana", background="#3a1c1c", font=(FONT, 11, "bold"))
        tree.tag_configure("member", foreground="#bcd0ff", background="#1a2138")
        tree.tag_configure("honmei", foreground=HONMEI)
        tree.tag_configure("chuuana", foreground=CHUUANA)
        tree.tag_configure("ooana", foreground=OOANA)
        vs = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vs.set)
        tree.pack(side="left", fill="both", expand=True, padx=(14, 0), pady=6)
        vs.pack(side="left", fill="y", padx=(0, 8), pady=6)

        state = {"results": [], "races": [], "row2race": {}}

        def load_races():
            iso = date_box.get()
            yymmdd = iso[2:].replace("-", "")
            races = program.parse_program(yymmdd)
            state["races"] = races
            venues = sorted({r["venue"] for r in races})
            venue_box["values"] = [f"{v}:{self.pred.venue_names.get(v, v)}" for v in venues]
            if venues:
                venue_box.current(0)
            return races

        def race_head(s, cond):
            w, d, wn, wv = cond
            return (f'{s["race"]}R  【{s["kind"]}】  '
                    f'1着候補→{s["best"]}号艇 / 最下位→{s["worst"]}号艇   '
                    f'［{w}/{d}/風{wn}/波{wv}］')

        def fill_children(rid, s):
            for ch in tree.get_children(rid):
                tree.delete(ch)
            # まず出走メンバー（各号艇の選手名＋1着率＋印）
            full = s["full"]
            strg = full["strength"]
            best = int(strg.argmax())
            worst = int(full["p_last"].argmax())
            for i, info in enumerate(full["infos"]):
                mk = ""
                if i == best:
                    mk = "◎1着候補"
                if i == worst:
                    mk = (mk + " ▽最下位") if mk else "▽最下位候補"
                tree.insert(rid, "end", text=f"   👤 {i+1}号艇  {info['_name']}",
                            tags=("member",),
                            values=(f"1着 {strg[i]*100:.0f}%", "", "", mk))
            # 買い目は3連単・本命のみ（上位10点）。大穴/中穴は的中率が低いため非表示。
            for j, c in enumerate(s["full"]["combos"][:10]):
                tree.insert(rid, "end", text=f"   ◎ 本命{j+1}", tags=("honmei",),
                            values=(c["combo"], f'{c["prob"]*100:.1f}%',
                                    f'約{c["payout"]:,}円', ""))
            tree.item(rid, open=True)

        def run():
            try:
                races = state["races"] or load_races()
                vsel = venue_box.get()
                if not vsel:
                    messagebox.showinfo("確認", "レース場を選んでください。")
                    return
                vcode = int(vsel.split(":")[0])
                races = [r for r in races if r["venue"] == vcode]
                wind = float(wind_e.get() or 0)
                wave = float(wave_e.get() or 0)
            except Exception as e:
                messagebox.showerror("エラー", str(e))
                return
            if not races:
                messagebox.showinfo("確認", "その場のレースがありません。")
                return
            status.config(text=f"予想中... 0/{len(races)}")
            win.update()

            def prog(done, total):
                status.config(text=f"予想中... {done}/{total}")
                win.update()

            gcond = (w_box.get(), d_box.get(), wind, wave)
            res = self.pred.predict_program(races, *gcond[:2], gcond[2], gcond[3],
                                            progress=prog, date_iso=date_box.get())
            state["results"] = res
            state["races_f"] = races
            state["cond"] = [tuple(gcond) for _ in res]   # 全レース同条件で初期化
            state["row2race"] = {}
            state["race_rows"] = {}
            tree.delete(*tree.get_children())
            n_ana = 0
            for i, s in enumerate(res):
                ana = s["kind"].startswith("波乱")
                n_ana += ana
                rid = tree.insert("", "end", text=race_head(s, state["cond"][i]),
                                  tags=("race_ana" if ana else "race",),
                                  values=("", "", "",
                                          ("⚠ 大穴が狙えるかも" if ana else "")))
                state["row2race"][rid] = i
                state["race_rows"][i] = rid
                fill_children(rid, s)
            vn = res[0]["venue_name"] if res else ""
            status.config(text=(f"完了！ {vn} {len(res)}レース ／ 大穴注意 {n_ana}件  "
                                "（レースを選び条件を変えて『選択Rに適用』／"
                                "見出しダブルクリックで詳細）"))

        def apply_one():
            """選択中のレースだけ、今の条件欄の天候/風/波で再予想する。"""
            if not state["results"]:
                return
            sel = tree.focus()
            if not sel:
                messagebox.showinfo("確認", "条件を変えたいレースを一覧で選んでください。")
                return
            rrow = sel if sel in state["row2race"] else tree.parent(sel)
            if rrow not in state["row2race"]:
                return
            i = state["row2race"][rrow]
            try:
                wind = float(wind_e.get() or 0); wave = float(wave_e.get() or 0)
            except ValueError:
                messagebox.showwarning("入力エラー", "風速・波高は数字で。")
                return
            cond = (w_box.get(), d_box.get(), wind, wave)
            race = state["races_f"][i]
            s = self.pred.predict_program([race], *cond[:2], cond[2], cond[3],
                                          date_iso=date_box.get())[0]
            state["results"][i] = s
            state["cond"][i] = cond
            ana = s["kind"].startswith("波乱")
            tree.item(rrow, text=race_head(s, cond),
                      tags=("race_ana" if ana else "race"),
                      values=("", "", "", ("⚠ 大穴が狙えるかも" if ana else "")))
            fill_children(rrow, s)
            tree.see(rrow)
            status.config(text=f"{s['race']}R を [{cond[0]}/{cond[1]}/風{cond[2]}/波{cond[3]}] "
                               "で再予想しました。")

        def on_select(_e=None):
            """レース選択時、その条件を条件欄に反映（個別編集しやすく）。"""
            if not state.get("cond"):
                return
            sel = tree.focus()
            rrow = sel if sel in state["row2race"] else tree.parent(sel)
            if rrow not in state["row2race"]:
                return
            w, d, wn, wv = state["cond"][state["row2race"][rrow]]
            w_box.set(w); d_box.set(d)
            wind_e.delete(0, "end"); wind_e.insert(0, str(wn))
            wave_e.delete(0, "end"); wave_e.insert(0, str(wv))

        def on_open(_e=None):
            sel = tree.focus()
            if not sel:
                return
            # 子(買い目)を押した場合は親レースへ
            race_row = sel if sel in state["row2race"] else tree.parent(sel)
            if race_row not in state["row2race"]:
                return
            s = state["results"][state["row2race"][race_row]]
            for box_i, code in enumerate(s["ids"]):
                self.racer_vars[box_i].set(str(code))
            for idx, (k, _v) in enumerate(self.pred.venue_names.items()):
                if k == s["venue"]:
                    self.venue_box.current(idx)
                    break
            self.weather_box.set(w_box.get())
            self.winddir_box.set(d_box.get())
            self.wind_var.set(wind_e.get())
            self.wave_var.set(wave_e.get())
            self.on_predict()
            self.lift()

        def save_csv():
            if not state["results"]:
                messagebox.showinfo("CSV保存", "先に予想を実行してください。")
                return
            iso = date_box.get()
            vn = state["results"][0]["venue_name"]
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"予想_{iso}_{vn}.csv")
            import csv
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["場", "R", "判定", "天候", "風向", "風速", "波高",
                             "1着候補", "最下位候補",
                             "種別", "順位", "買い目", "的中率", "想定配当"])
                for i, s in enumerate(state["results"]):
                    cw, cd, cwn, cwv = state["cond"][i]
                    for label, key in (("本命", "honmei"), ("中穴", "chuuana"),
                                       ("大穴", "ooana")):
                        for j, c in enumerate(s["full"][key]):
                            wr.writerow([
                                s["venue_name"], s["race"], s["kind"],
                                cw, cd, cwn, cwv,
                                f'{s["best"]}号艇', f'{s["worst"]}号艇',
                                label, j + 1, c["combo"],
                                f'{c["prob"]*100:.1f}%', f'約{c["payout"]:,}円'])
            messagebox.showinfo("CSV保存", f"保存しました:\n{path}")

        tk.Label(win, text="※ レース見出しをダブルクリックすると、その6名をメイン画面に"
                 "入れて詳細予想します。", bg=BG, fg=SUB, font=(FONT, 9)).pack(
                 side="bottom", anchor="w", padx=14, pady=4)
        tree.bind("<Double-1>", on_open)
        tree.bind("<<TreeviewSelect>>", on_select)
        date_box.bind("<<ComboboxSelected>>", lambda e: load_races())

        load_races()

    # ---------------------------------------------------- 予想実行
    def on_predict(self):
        ids = []
        for i, v in enumerate(self.racer_vars):
            s = v.get().strip()
            if not s.isdigit():
                messagebox.showwarning("入力エラー",
                                       f"{i+1}号艇の登録番号を数字で入力してください。")
                return
            ids.append(int(s))
        try:
            wind = float(self.wind_var.get())
            wave = float(self.wave_var.get())
        except ValueError:
            messagebox.showwarning("入力エラー", "風速・波高は数字で入力してください。")
            return

        venue = self._venue()
        weather, wdir = self.weather_box.get(), self.winddir_box.get()
        exhibit = self._exhibit_list()

        # 枠なり と 進入予想 の両方を計算（どちらも知りたい、というご要望）
        courses_sn = self.pred.predict_courses(venue, ids)
        r_yk = self.pred.predict(venue, ids, weather, wdir, wind, wave,
                                 courses=None, exhibit=exhibit)
        r_sn = self.pred.predict(venue, ids, weather, wdir, wind, wave,
                                 courses=courses_sn, exhibit=exhibit)
        use_sn = self.shinnyuu_var.get()
        r = r_sn if use_sn else r_yk

        # 進入の表示文字列
        maezuke = courses_sn != [1, 2, 3, 4, 5, 6]
        order = "-".join(str(courses_sn[i]) for i in range(6))
        if maezuke:
            movers = [f"{i+1}号艇→{courses_sn[i]}コース"
                      for i in range(6) if courses_sn[i] != i + 1]
            sn_txt = f"前づけ予想あり（{ '、'.join(movers) }）"
        else:
            sn_txt = "枠なり（前づけの兆候なし）"
        best_yk = int(r_yk["strength"].argmax()) + 1
        best_sn = int(r_sn["strength"].argmax()) + 1
        mark = "【進入予想モード】" if use_sn else "【枠なりモード】"
        info = (f"{mark}  予想進入：{order}（{sn_txt}）   "
                f"1着候補→ 枠なり:{best_yk}号艇 / 進入予想:{best_sn}号艇")
        if exhibit is not None:
            info += "   ※展示タイム反映中"
        self.info_bar.config(
            text=info, fg=(OOANA if maezuke else SUB))

        self._show_boats(r)
        self._show_tips(r)

    def _show_boats(self, r):
        for w in self.boat_area.winfo_children():
            w.destroy()
        colors = ["#ffffff", "#9aa0a6", "#ff5a5a", "#5a86ff", "#ffd400", "#3fdd55"]
        win = r["strength"]
        last = r["p_last"]
        best = int(win.argmax())
        worst = int(last.argmax())
        for i in range(6):
            row = tk.Frame(self.boat_area, bg=PANEL)
            row.pack(fill="x", pady=4)
            tag = ""
            if i == best:
                tag = "  ◎ 1着本命"
            if i == worst:
                tag += "  ▽ 最下位候補"
            head = tk.Label(row, bg=PANEL, fg=colors[i], font=(FONT, 11, "bold"),
                            text=f"{i+1}号艇  {r['infos'][i]['_name']}{tag}", anchor="w")
            head.pack(anchor="w")
            inf = r["infos"][i]
            grade = ["", "B2", "B1", "A2", "A1"][int(inf["grade_num"])]
            local = f"{inf['rv_win']*100:.0f}%" if inf.get("_rv_found") else "—"
            sub = tk.Label(
                row, bg=PANEL, fg=SUB, font=(FONT, 9), anchor="w",
                text=(f"{grade} 全国{inf['win_nat']:.2f}  "
                      f"{i+1}コース勝率{inf['rc_win']*100:.0f}%  "
                      f"当地{local}  "
                      f"平均ST{inf['r_avg_st']:.2f}  "
                      f"直近{inf['r_recent_win']*100:.0f}%"))
            sub.pack(anchor="w")
            # 戦闘力（総合＝地力 / レース別＝この枠での1着力）
            pw = r["power"][i]
            tk.Label(
                row, bg=PANEL, fg="#ffd400", font=(FONT, 9, "bold"), anchor="w",
                text=(f"⚔ 戦闘力  総合 {pw['overall']}/100   "
                      f"レース別 {pw['race_score']}/100（1着力 {pw['race_pct']:.0f}%）")
            ).pack(anchor="w")
            tk.Label(
                row, bg=PANEL, fg="#7fdfff", font=(FONT, 9), anchor="w",
                text=(f"🎯 単勝 買い基準 {pw['tansho_be']:.2f}倍以上なら買い得"
                      "（実オッズがこれ以上のとき）")
            ).pack(anchor="w")
            # 1着バー
            self._bar(row, "1着", win[i], ACCENT)
            # 最下位バー
            self._bar(row, "最下位", last[i], "#ff8c8c")

    def _bar(self, parent, label, val, color):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x")
        tk.Label(f, text=label, bg=PANEL, fg=SUB, font=(FONT, 9),
                 width=6, anchor="w").pack(side="left")
        canvas = tk.Canvas(f, height=14, bg="#26345c", highlightthickness=0)
        canvas.pack(side="left", fill="x", expand=True, padx=(2, 6))
        canvas.update_idletasks()
        w = max(canvas.winfo_width(), 260)
        canvas.create_rectangle(0, 0, w * min(val, 1.0), 14, fill=color, width=0)
        tk.Label(f, text=f"{val*100:4.1f}%", bg=PANEL, fg=TEXT,
                 font=(FONT, 9, "bold"), width=6).pack(side="left")

    def _show_tips(self, r):
        for w in self.tip_area.winfo_children():
            w.destroy()

        def block(title, items, color, note=""):
            tk.Label(self.tip_area, text=title, bg=PANEL, fg=color,
                     font=(FONT, 12, "bold")).pack(anchor="w", pady=(8, 0))
            if note:
                tk.Label(self.tip_area, text=note, bg=PANEL, fg=SUB,
                         font=(FONT, 9)).pack(anchor="w")
            for c in items:
                line = tk.Frame(self.tip_area, bg=PANEL)
                line.pack(fill="x", pady=1)
                tk.Label(line, text=f"  {c['combo']}", bg=PANEL, fg=TEXT,
                         font=("Consolas", 13, "bold"), width=9, anchor="w").pack(side="left")
                tk.Label(line, text=f"的中率 {c['prob']*100:5.1f}%", bg=PANEL,
                         fg=SUB, font=(FONT, 10)).pack(side="left", padx=6)
                tk.Label(line, text=f"想定配当 約{c['payout']:,}円", bg=PANEL,
                         fg=color, font=(FONT, 10)).pack(side="left")
                if c.get("be_odds"):
                    tk.Label(line, text=f"｜{c['be_odds']:.1f}倍〜買い", bg=PANEL,
                             fg="#7fdfff", font=(FONT, 10)).pack(side="left", padx=4)

        # 荒れ度メーター（1号艇の1着率を見せ方として使う）
        self._arare_meter(self.tip_area, float(r["strength"][0]))
        # 狙い目アドバイス（番組から番号/レース名が分かる時だけ・参考）
        advs = [race_advice(self._race_no, self._race_title)]
        # 鳴門専用：風向×風速の検証済みアドバイス（今の条件から）
        if self.naruto:
            advs.append(naruto_wind_advice(self.winddir_box.get(), self.wind_var.get()))
        for adv in advs:
            if adv:
                tk.Label(self.tip_area, text=adv, bg="#101a30", fg="#9fe0ff",
                         font=(FONT, 10), wraplength=460, justify="left",
                         padx=8, pady=5).pack(fill="x", pady=(2, 4))
        # 券種別おすすめ（3連単/3連複/2連単/2連複 各3点）＋おすすめ券種
        rec, bt = r.get("recommend"), r.get("bet_types")
        if rec and bt:
            tk.Label(self.tip_area,
                     text=f"🏆 おすすめ券種：{rec['type']}（自信度 {rec['conf']*100:.0f}%）",
                     bg="#163a27", fg="#a9f5b0", font=(FONT, 12, "bold"),
                     anchor="w", padx=8, pady=5).pack(fill="x", pady=(6, 1))
            tk.Label(self.tip_area, text="　" + rec["reason"], bg=PANEL, fg=SUB,
                     font=(FONT, 9), wraplength=440, justify="left",
                     anchor="w").pack(fill="x")
            cols = {"3連単": OOANA, "3連複": HONMEI, "2連単": CHUUANA, "2連複": ACCENT}
            for t in ["3連単", "3連複", "2連単", "2連複"]:
                mark = " ◀ いちおし" if t == rec["type"] else ""
                block(f"【{t}】各3点{mark}", bt[t], cols[t])
        # 3連単・本命のみ（確率が高い上位10点）。大穴/中穴は的中率が低いため非表示。
        block("◎ 3連単 本命（上位10点）", r["combos"][:10], HONMEI)

        # 大穴判定コメント
        self.comment.config(text=self._ooana_comment(r))

    def _arare_meter(self, parent, p1):
        """1号艇の1着率(p1)を『荒れ度』として視覚化。低い=荒れ, 高い=堅い。"""
        if p1 >= 0.55:
            cat, col = "堅い（手堅い決着）", "#3fdd55"
        elif p1 >= 0.40:
            cat, col = "やや混戦", "#ffd400"
        else:
            cat, col = "荒れやすい（高配当に注意）", "#ff6b6b"
        fr = tk.Frame(parent, bg=PANEL)
        fr.pack(fill="x", pady=(2, 6))
        tk.Label(fr, text="荒れ度メーター", bg=PANEL, fg=ACCENT,
                 font=(FONT, 12, "bold")).pack(anchor="w")
        row = tk.Frame(fr, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, text="荒れ", bg=PANEL, fg="#ff6b6b", font=(FONT, 9)).pack(side="left")
        canvas = tk.Canvas(row, height=16, bg="#26345c", highlightthickness=0)
        canvas.pack(side="left", fill="x", expand=True, padx=4)
        canvas.update_idletasks()
        w = max(canvas.winfo_width(), 240)
        frac = min(max((p1 - 0.25) / 0.40, 0.0), 1.0)  # 0=荒れ(左) 〜 1=堅い(右)
        x = w * frac
        canvas.create_rectangle(x - 4, 0, x + 4, 16, fill=col, width=0)
        tk.Label(row, text="堅い", bg=PANEL, fg="#3fdd55", font=(FONT, 9)).pack(side="left")
        tk.Label(fr, text=f"判定：{cat}　（1号艇の1着率 {p1*100:.0f}%）",
                 bg=PANEL, fg=col, font=(FONT, 10, "bold")).pack(anchor="w")

    @staticmethod
    def _ooana_comment(r):
        win = r["strength"]
        top = win.max()
        best = int(win.argmax()) + 1
        # 大穴の最有力（=大穴帯で一番確率の高い目）
        oo = r["ooana"][0] if r["ooana"] else None
        msg = ""
        if top < 0.40:
            msg = ("⚠ 大穴注意レース！ 1号艇クラスの絶対的本命が不在で、"
                   "波乱（高配当）が出やすい組み合わせです。")
            if oo:
                msg += f"  特に【{oo['combo']}】(想定約{oo['payout']:,}円)は一発の可能性アリ。"
        elif top < 0.55:
            msg = (f"○ やや混戦。{best}号艇が中心ですが崩れる余地あり。"
                   "中穴まで広げる価値があります。")
            if oo:
                msg += f"  大穴なら【{oo['combo']}】に妙味。"
        else:
            msg = (f"◎ 堅いレース。{best}号艇の1着が濃厚（{top*100:.0f}%）。"
                   "本命中心で手堅く。大穴向きではありません。")
        return msg


if __name__ == "__main__":
    App().mainloop()
