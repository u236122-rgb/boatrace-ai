# -*- coding: utf-8 -*-
"""
web.py  ―  競艇AI予想 スマホ用Webサーバ（追加インストール不要・標準ライブラリのみ）
============================================================
PCでこのサーバを起動し、同じWi-Fiのスマホのブラウザで
    http://<PCのIP>:8000
を開くと、スマホから予想できる。

  起動:  python web.py
  （事前に build_model.py でモデルを作成しておくこと）
"""
import os
import json
import re
import time
import hmac
import base64
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from core import Predictor, race_advice
import program

PRED = Predictor()
PORT = 8000

# ★外部公開する場合のアクセスパスワード（スマホで初回に入力）。
#   コードには書かず、環境変数 KYOTEI_WEB_PASSWORD で渡す。
#   未設定ならパスワードなし（同じWi-Fi内だけで使う時向け）。
#   tunnel.py（外部公開）は未設定だと起動しない。
PASSWORD = os.environ.get("KYOTEI_WEB_PASSWORD", "")
USERNAME = os.environ.get("KYOTEI_WEB_USER", "user")   # ログイン画面のユーザー名


def racer_name(rid):
    try:
        if rid in PRED.racer_global.index:
            return str(PRED.racer_global.loc[rid, "name"])
    except Exception:
        pass
    return f"No.{rid}"


def jget(obj):
    """numpy型をJSONに変換。"""
    import numpy as np
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(str(type(obj)))


def _tip(lst):
    return [{"combo": c["combo"], "prob": c["prob"], "payout": c["payout"],
             "be_odds": c.get("be_odds")}
            for c in lst]


def do_program_predict(body):
    """場を指定して、その日の全レースをまとめ予想。"""
    ymd = str(body.get("date", ""))
    if not re.fullmatch(r"\d{6}", ymd):
        raise ValueError("日付が不正です")
    venue = int(body["venue"])
    weather = body.get("weather", "晴")
    winddir = body.get("winddir", "無風")
    wind = float(body.get("wind", 0) or 0)
    wave = float(body.get("wave", 0) or 0)

    races = [r for r in program.parse_program(ymd) if r["venue"] == venue]
    res = PRED.predict_program(races, weather, winddir, wind, wave,
                               date_iso=program.yymmdd_to_iso(ymd))
    out = []
    for s in res:
        full = s["full"]
        strg = full["strength"]
        best = int(strg.argmax())
        worst = int(full["p_last"].argmax())
        boats = [{"boat": i + 1, "name": full["infos"][i]["_name"],
                  "strength": float(strg[i]),
                  "is_best": i == best, "is_worst": i == worst}
                 for i in range(6)]
        out.append({
            "race": s["race"], "kind": s["kind"],
            "best": s["best"], "worst": s["worst"], "boats": boats,
            "honmei": _tip(full["honmei"]), "chuuana": _tip(full["chuuana"]),
            "ooana": _tip(full["ooana"]),
        })
    return {"venue_name": PRED.venue_names.get(venue, str(venue)), "races": out}


def do_predict(body):
    venue = int(body["venue"])
    ids = [int(x) for x in body["ids"]]
    weather = body.get("weather", "晴")
    winddir = body.get("winddir", "無風")
    wind = float(body.get("wind", 0) or 0)
    wave = float(body.get("wave", 0) or 0)
    use_sn = bool(body.get("shinnyuu", False))
    exhibit = body.get("exhibit") or None

    courses_sn = PRED.predict_courses(venue, ids)
    r_yk = PRED.predict(venue, ids, weather, winddir, wind, wave,
                        courses=None, exhibit=exhibit)
    r_sn = PRED.predict(venue, ids, weather, winddir, wind, wave,
                        courses=courses_sn, exhibit=exhibit)
    r = r_sn if use_sn else r_yk

    best = int(r["strength"].argmax())
    worst = int(r["p_last"].argmax())
    grades = ["", "B2", "B1", "A2", "A1"]
    pw = r["power"]
    boats = []
    for i in range(6):
        inf = r["infos"][i]
        boats.append({
            "boat": i + 1,
            "name": inf["_name"],
            "strength": float(r["strength"][i]),
            "pw_all": int(pw[i]["overall"]),
            "pw_race": int(pw[i]["race_score"]),
            "tansho_be": float(pw[i]["tansho_be"]),
            "last": float(r["p_last"][i]),
            "grade": grades[int(inf["grade_num"])],
            "win_nat": float(inf["win_nat"]),
            "rc_win": float(inf["rc_win"]),
            "rv_win": float(inf["rv_win"]),
            "rv_found": bool(inf.get("_rv_found")),
            "avg_st": float(inf["r_avg_st"]),
            "recent": float(inf["r_recent_win"]),
            "course": int(courses_sn[i]),
            "is_best": i == best,
            "is_worst": i == worst,
        })

    def tip(lst):
        return [{"combo": c["combo"], "prob": c["prob"], "payout": c["payout"],
                 "be_odds": c.get("be_odds")}
                for c in lst]

    top = float(r["strength"].max())
    kind = "堅い" if top >= 0.55 else ("やや混戦" if top >= 0.40 else "波乱(大穴注意)")
    maezuke = courses_sn != [1, 2, 3, 4, 5, 6]
    return {
        "boats": boats,
        "best": best + 1, "worst": worst + 1,
        "honmei": tip(r["honmei"]), "chuuana": tip(r["chuuana"]),
        "ooana": tip(r["ooana"]),
        "kind": kind,
        "courses": courses_sn,
        "maezuke": maezuke,
        "best_yk": int(r_yk["strength"].argmax()) + 1,
        "best_sn": int(r_sn["strength"].argmax()) + 1,
        "exhibit_used": bool(r.get("exhibit_used")),
        "venue_name": r["venue_name"],
        "advice": race_advice(body.get("race_no"), body.get("title", "")),
        "bet_types": {t: _tip(v) for t, v in r["bet_types"].items()},
        "recommend": {"type": r["recommend"]["type"],
                      "reason": r["recommend"]["reason"],
                      "conf": float(r["recommend"]["conf"])},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _auth_ok(self):
        if not PASSWORD:
            return True
        h = self.headers.get("Authorization", "")
        if h.startswith("Basic "):
            try:
                dec = base64.b64decode(h[6:]).decode("utf-8")
                user, _, pw = dec.partition(":")
                # タイミング攻撃を避けて定数時間で照合
                ok_pw = hmac.compare_digest(pw, PASSWORD)
                ok_user = (not USERNAME) or hmac.compare_digest(user, USERNAME)
                if ok_pw and ok_user:
                    return True
            except Exception:
                pass
        time.sleep(1.0)   # 総当たりを遅くする簡易スロットル
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Kyotei AI"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _send(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, "application/json; charset=utf-8",
                   json.dumps(obj, ensure_ascii=False, default=jget).encode("utf-8"))

    def do_GET(self):
        if not self._auth_ok():
            return
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            return
        if u.path == "/api/init":
            dates = program.list_program_dates()
            self._json({
                "dates": [{"yymmdd": d, "iso": program.yymmdd_to_iso(d)}
                          for d in dates[:60]],
                "venues": [{"code": k, "name": v}
                           for k, v in PRED.venue_names.items()],
                "weather": PRED.weather_list,
                "winddir": PRED.winddir_list,
                "info": {"n_races": PRED.n_races,
                         "acc": round(PRED.valid_acc * 100, 1),
                         "max_date": PRED.data_max_date},
            })
            return
        if u.path == "/api/program":
            q = parse_qs(u.query)
            ymd = q.get("date", [""])[0]
            # 日付は数字6桁のみ許可（パストラバーサル防止）
            if not re.fullmatch(r"\d{6}", ymd):
                self._json({"error": "日付の形式が不正です"}, 400)
                return
            try:
                venue = int(q.get("venue", ["0"])[0])
            except ValueError:
                self._json({"error": "場が不正です"}, 400)
                return
            try:
                races = program.parse_program(ymd)
            except Exception as e:
                self._json({"error": str(e)}, 400)
                return
            out = []
            for r in races:
                if r["venue"] != venue:
                    continue
                out.append({
                    "race": r["race"], "title": r.get("title", ""),
                    "racers": [{"boat": x["boat"], "id": x["racer_id"],
                                "name": racer_name(x["racer_id"])}
                               for x in sorted(r["racers"], key=lambda b: b["boat"])],
                })
            self._json({"races": out})
            return
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if not self._auth_ok():
            return
        u = urlparse(self.path)
        ln = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(ln) or b"{}")
        if u.path == "/api/predict":
            try:
                self._json(do_predict(body))
            except Exception as e:
                self._json({"error": str(e)}, 400)
            return
        if u.path == "/api/program_predict":
            try:
                self._json(do_program_predict(body))
            except Exception as e:
                self._json({"error": str(e)}, 400)
            return
        self._send(404, "text/plain", b"not found")


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


PAGE = r"""<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>競艇AI予想</title>
<style>
:root{--bg:#0f1626;--panel:#1b2540;--accent:#36c2ff;--honmei:#ffd43b;
--chuuana:#69db7c;--ooana:#ff6b6b;--text:#e9eefb;--sub:#9fb0d0;--field:#eaf2ff;--ft:#10203a;}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
body{margin:0;background:var(--bg);color:var(--text);
font-family:"Hiragino Kaku Gothic ProN","Yu Gothic",sans-serif;font-size:15px;}
.wrap{max-width:560px;margin:0 auto;padding:12px;}
h1{font-size:18px;color:var(--accent);margin:4px 0 2px;}
.muted{color:var(--sub);font-size:12px;}
.card{background:var(--panel);border-radius:12px;padding:12px;margin:10px 0;}
label{display:block;color:var(--sub);font-size:12px;margin-bottom:3px;}
select,input{width:100%;padding:10px;border-radius:8px;border:0;background:var(--field);
color:var(--ft);font-size:16px;}
.row{display:flex;gap:8px;}.row>div{flex:1;}
.btn{width:100%;padding:13px;border:0;border-radius:10px;font-size:16px;font-weight:bold;
background:var(--accent);color:#06223a;margin-top:8px;}
.btn.sub{background:#2a3a63;color:var(--text);font-weight:normal;}
.btn.gold{background:var(--honmei);color:#3a2e00;}
.boats6{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;}
.boats6 .b{background:#101a30;border-radius:8px;padding:6px;text-align:center;}
.boats6 .b label{font-size:11px;}
.boats6 .b input{padding:7px;font-size:16px;text-align:center;}
.nm{font-size:11px;color:var(--sub);min-height:14px;margin-top:3px;}
.chk{display:flex;align-items:center;gap:8px;margin-top:10px;color:var(--text);font-size:14px;}
.chk input{width:auto;transform:scale(1.3);}
.info{background:#101a30;border-radius:8px;padding:8px;font-size:12px;margin:8px 0;}
.boatrow{display:flex;align-items:center;gap:8px;padding:7px 4px;border-bottom:1px solid #243154;}
.bn{width:62px;font-weight:bold;font-size:14px;}
.bar{flex:1;background:#0c1322;border-radius:6px;height:18px;position:relative;overflow:hidden;}
.bar>span{position:absolute;left:0;top:0;bottom:0;background:var(--accent);}
.bv{width:42px;text-align:right;font-size:13px;}
.bsub{font-size:10px;color:var(--sub);}
.tag{font-size:11px;padding:1px 6px;border-radius:4px;}
.tag.b{background:#143b52;color:var(--accent);}.tag.w{background:#3a1c1c;color:var(--ooana);}
.tips h3{margin:10px 0 4px;font-size:14px;}
.tips .h{color:var(--honmei);}.tips .c{color:var(--chuuana);}.tips .o{color:var(--ooana);}
.tip{display:flex;justify-content:space-between;padding:5px 6px;border-bottom:1px solid #243154;font-size:14px;}
.tip .pay{color:var(--sub);font-size:12px;}
.cmt{background:#101a30;border-left:4px solid var(--ooana);padding:8px;border-radius:6px;margin-top:8px;font-size:13px;}
.meter{margin:10px 0 6px;}
.mbar{position:relative;height:14px;background:#26345c;border-radius:7px;margin:4px 0;}
.mbar>span{position:absolute;top:0;width:8px;height:14px;border-radius:4px;transform:translateX(-4px);}
.disc{font-size:11px;color:#d8b24a;margin-top:8px;}
.rec{font-size:11px;color:var(--sub);margin:2px 0;}
.tabs{display:flex;gap:6px;margin-bottom:6px;}
.tabs button{flex:1;padding:8px;border:0;border-radius:8px;background:#2a3a63;color:var(--text);font-size:13px;}
.tabs button.on{background:var(--accent);color:#06223a;font-weight:bold;}
.mrace{border:1px solid #243154;border-radius:8px;margin:6px 0;overflow:hidden;}
.mrace.ana{border-color:#5a2a2a;}
.mhead{padding:9px 10px;background:#243154;cursor:pointer;display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.mrace.ana .mhead{background:#3a1c1c;}
.mhead .kind{font-size:11px;color:var(--sub);}
.mhead .sub{font-size:11px;color:var(--sub);margin-left:auto;}
.mhead .anatag{font-size:11px;color:var(--ooana);}
.mbody{display:none;padding:8px 10px;}
.mrace.open .mbody{display:block;}
.mb{padding:4px 2px;border-bottom:1px solid #1a2336;font-size:14px;}
.mb .sub{color:var(--sub);font-size:12px;}
.ex6{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px;}
.ex6 input{padding:11px 4px;font-size:16px;text-align:center;width:100%;}
.ex6 label{font-size:12px;color:var(--sub);text-align:center;display:block;margin-bottom:3px;}
#exDetails summary{list-style:none;background:#2a3a63;color:var(--text);padding:11px;
  border-radius:8px;font-size:14px;text-align:center;cursor:pointer;font-weight:bold;}
#exDetails summary::-webkit-details-marker{display:none;}
#exDetails[open] summary{background:var(--accent);color:#06223a;}
</style></head>
<body><div class="wrap">
<h1>🚤 競艇AI予想</h1>
<div class="muted" id="hdr">読み込み中…</div>

<div class="card">
  <div class="tabs">
    <button id="tabLoad" class="on" onclick="setTab('load')">番組から</button>
    <button id="tabManual" onclick="setTab('manual')">番号入力</button>
    <button id="tabMatome" onclick="setTab('matome')">当日まとめ</button>
  </div>

  <div class="row">
    <div><label>場</label><select id="venue"></select></div>
    <div><label>日付</label><select id="date"></select></div>
  </div>

  <div id="loadPane">
    <div class="row" style="margin-top:8px;">
      <div><label>レース</label><select id="race"></select></div>
      <div style="display:flex;align-items:flex-end">
        <button class="btn sub" style="margin:0" onclick="loadRace()">この場・R を読み込む</button>
      </div>
    </div>
  </div>

  <div id="boats6" class="boats6" style="margin-top:10px;"></div>

  <div class="row" style="margin-top:10px;">
    <div><label>天候</label><select id="weather"></select></div>
    <div><label>風向</label><select id="winddir"></select></div>
  </div>
  <div class="row" style="margin-top:8px;">
    <div><label>風速(m)</label><input id="wind" type="number" value="3" inputmode="numeric"></div>
    <div><label>波高(cm)</label><input id="wave" type="number" value="3" inputmode="numeric"></div>
  </div>

  <div id="singleArea">
    <label class="chk"><input type="checkbox" id="shinnyuu">進入予想を反映（前づけを考慮）</label>
    <details id="exDetails" style="margin-top:8px">
      <summary>▸ 直前情報（展示タイム）を入力 ＝予想に反映</summary>
      <div class="ex6" id="ex6"></div>
    </details>
    <button class="btn" onclick="predict()">🔮 予想する</button>
  </div>
  <div id="matomeArea" style="display:none">
    <button class="btn gold" onclick="matomePredict()">📋 この場の全レースを予想</button>
  </div>
</div>

<div id="result"></div>

</div>
<script>
let INIT=null;
let LOADED=null;   // 番組から読み込んだレース文脈（狙い目アドバイス用）
const $=id=>document.getElementById(id);

async function boot(){
  INIT=await (await fetch('/api/init')).json();
  $('hdr').textContent=`学習 ${INIT.info.n_races.toLocaleString()}レース / 本命的中率 ${INIT.info.acc}% / 最終 ${INIT.info.max_date}`;
  fill($('venue'),INIT.venues.map(v=>[v.code,`${v.code}: ${v.name}`]));
  fill($('date'),INIT.dates.map(d=>[d.yymmdd,d.iso]));
  fill($('race'),Array.from({length:12},(_,i)=>[i+1,(i+1)+'R']));
  fill($('weather'),INIT.weather.map(w=>[w,w]));
  fill($('winddir'),INIT.winddir.map(w=>[w,w]));
  // 既定=びわこ
  $('venue').value=11;
  buildBoats();
}
function fill(sel,pairs){sel.innerHTML='';for(const [v,t] of pairs){const o=document.createElement('option');o.value=v;o.textContent=t;sel.appendChild(o);}}
function buildBoats(){
  const cols=["#ffffff","#9aa0a6","#ff5a5a","#5a86ff","#ffd400","#3fdd55"];
  let h='';
  for(let i=0;i<6;i++){
    h+=`<div class="b"><label style="color:${cols[i]}">${i+1}号艇</label>
    <input id="r${i}" inputmode="numeric" oninput="showName(${i});LOADED=null;">
    <div class="nm" id="n${i}"></div></div>`;
  }
  $('boats6').innerHTML=h;
  let eh='';
  for(let i=0;i<6;i++){eh+=`<div><label>${i+1}号艇</label><input id="ex${i}" inputmode="decimal" placeholder="-"></div>`;}
  $('ex6').innerHTML=eh;
}
let NAMECACHE={};
function showName(i){
  const v=$('r'+i).value.trim();
  $('n'+i).textContent = v && NAMECACHE[v] ? NAMECACHE[v] : (v?'…':'');
}
function setTab(t){
  $('tabLoad').classList.toggle('on',t==='load');
  $('tabManual').classList.toggle('on',t==='manual');
  $('tabMatome').classList.toggle('on',t==='matome');
  $('loadPane').style.display = t==='load'?'block':'none';
  $('boats6').style.display = t==='matome'?'none':'grid';
  $('singleArea').style.display = t==='matome'?'none':'block';
  $('matomeArea').style.display = t==='matome'?'block':'none';
  $('result').innerHTML='';
}
async function matomePredict(){
  const body={date:$('date').value,venue:+$('venue').value,weather:$('weather').value,
    winddir:$('winddir').value,wind:+$('wind').value,wave:+$('wave').value};
  $('result').innerHTML='<div class="card">全レース予想中…（数秒）</div>';
  const r=await (await fetch('/api/program_predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  if(r.error){$('result').innerHTML='<div class="card">エラー: '+r.error+'</div>';return;}
  if(!r.races||!r.races.length){$('result').innerHTML='<div class="card">その日のその場は開催がありません。</div>';return;}
  renderMatome(r);
}
function renderMatome(r){
  const cols=["#ffffff","#9aa0a6","#ff5a5a","#5a86ff","#ffd400","#3fdd55"];
  let h=`<div class="card"><div style="font-weight:bold;color:var(--accent);margin-bottom:6px">${r.venue_name}　全${r.races.length}レース（タップで展開）</div>`;
  for(const s of r.races){
    const ana=s.kind.indexOf('波乱')===0;
    h+=`<div class="mrace ${ana?'ana':''}">
      <div class="mhead" onclick="this.parentNode.classList.toggle('open')">
        <b>${s.race}R</b> <span class="kind">${s.kind}</span>
        <span class="sub">1着→${s.best} / 最下位→${s.worst}</span>
        ${ana?'<span class="anatag">⚠大穴</span>':''}
      </div><div class="mbody">`;
    for(const b of s.boats){
      let mk='';if(b.is_best)mk=' <span class="tag b">◎</span>';if(b.is_worst)mk+=' <span class="tag w">▽</span>';
      h+=`<div class="mb"><span style="color:${cols[b.boat-1]}">${b.boat}</span> ${b.name} <span class="sub">${Math.round(b.strength*100)}%</span>${mk}</div>`;
    }
    h+='<div class="tips">';
    h+=tipBlock('◎ 本命','h',s.honmei);
    h+=tipBlock('○ 中穴','c',s.chuuana);
    h+=tipBlock('★ 大穴','o',s.ooana);
    h+='</div></div></div>';
  }
  h+='</div>';
  $('result').innerHTML=h;
}
async function loadRace(){
  const ymd=$('date').value, venue=$('venue').value, race=+$('race').value;
  const j=await (await fetch(`/api/program?date=${ymd}&venue=${venue}`)).json();
  if(j.error){alert(j.error);return;}
  const r=(j.races||[]).find(x=>x.race===race);
  if(!r){alert('その日のその場・Rは番組にありません。');return;}
  for(const m of r.racers){
    $('r'+(m.boat-1)).value=m.id; NAMECACHE[m.id]=m.name; $('n'+(m.boat-1)).textContent=m.name;
  }
  LOADED={race:r.race,title:r.title||''};
  predict();
}
async function predict(){
  const ids=[];
  for(let i=0;i<6;i++){const v=$('r'+i).value.trim();if(!/^\d+$/.test(v)){alert((i+1)+'号艇の登録番号を入れてください');return;}ids.push(+v);}
  let exhibit=null,anyEx=false;const evs=[];
  for(let i=0;i<6;i++){const el=$('ex'+i);const v=el?el.value.trim():'';evs.push(v);if(v)anyEx=true;}
  if(anyEx)exhibit=evs;
  const body={venue:+$('venue').value,ids,weather:$('weather').value,winddir:$('winddir').value,
    wind:+$('wind').value,wave:+$('wave').value,shinnyuu:$('shinnyuu').checked,exhibit,
    race_no:LOADED?LOADED.race:null,title:LOADED?LOADED.title:''};
  $('result').innerHTML='<div class="card">予想中…</div>';
  const r=await (await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  if(r.error){$('result').innerHTML='<div class="card">エラー: '+r.error+'</div>';return;}
  render(r);
}
function render(r){
  let h='<div class="card">';
  const order=r.courses.join('-');
  const snTxt=r.maezuke?('前づけ予想あり'):'枠なり（前づけなし）';
  h+=`<div class="info">予想進入：<b>${order}</b>（${snTxt}）<br>
  1着候補→ 枠なり:<b>${r.best_yk}号艇</b> / 進入予想:<b>${r.best_sn}号艇</b>
  ${r.exhibit_used?'　※展示反映':''}</div>`;
  // 各艇
  const cols=["#ffffff","#9aa0a6","#ff5a5a","#5a86ff","#ffd400","#3fdd55"];
  for(const b of r.boats){
    const pct=Math.round(b.strength*100);
    let tags='';
    if(b.is_best)tags+=' <span class="tag b">◎1着本命</span>';
    if(b.is_worst)tags+=' <span class="tag w">▽最下位</span>';
    const local=b.rv_found?Math.round(b.rv_win*100)+'%':'—';
    h+=`<div class="boatrow"><div class="bn" style="color:${cols[b.boat-1]}">${b.boat}号艇</div>
    <div style="flex:1">
      <div>${b.name}${tags}</div>
      <div class="bsub">${b.grade} 全国${b.win_nat.toFixed(2)} ${b.boat}ｺｰｽ${Math.round(b.rc_win*100)}% 当地${local} ST${b.avg_st.toFixed(2)}</div>
      <div class="bsub" style="color:#ffd400">⚔戦闘力 総合${b.pw_all}/100 ・ レース別${b.pw_race}/100（1着力${pct}%）</div>
      <div class="bsub" style="color:#7fdfff">🎯単勝 買い基準 ${b.tansho_be.toFixed(2)}倍以上なら買い得</div>
      <div class="bar"><span style="width:${pct}%"></span></div>
    </div><div class="bv">${pct}%</div></div>`;
  }
  // 荒れ度メーター（1号艇の1着率を見せ方として使う）
  const p1=r.boats[0].strength;
  let acat,acol;
  if(p1>=0.55){acat='堅い（手堅い決着）';acol='#3fdd55';}
  else if(p1>=0.40){acat='やや混戦';acol='#ffd400';}
  else{acat='荒れやすい（高配当注意）';acol='#ff6b6b';}
  const frac=Math.min(Math.max((p1-0.25)/0.40,0),1)*100;
  h+=`<div class="meter"><div style="font-weight:bold;color:var(--accent)">荒れ度メーター</div>
    <div class="mbar"><span style="left:${frac}%;background:${acol}"></span></div>
    <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--sub)"><span>荒れ</span><span>堅い</span></div>
    <div style="color:${acol};font-weight:bold;font-size:13px">判定：${acat}（1号艇の1着率 ${Math.round(p1*100)}%）</div></div>`;
  // 狙い目アドバイス（番組から番号/レース名が分かる時だけ）
  if(r.advice){h+=`<div style="background:#101a30;color:#9fe0ff;font-size:12px;padding:7px 9px;border-radius:6px;margin:6px 0">${r.advice}</div>`;}
  // 券種別おすすめ（3連単/3連複/2連単/2連複 各3点）
  h+='<div class="tips">';
  if(r.recommend&&r.bet_types){
    h+=`<div class="rec">🏆 おすすめ券種：<b>${r.recommend.type}</b>（自信度 ${Math.round(r.recommend.conf*100)}%）<br>${r.recommend.reason}</div>`;
    const cls={'3連単':'o','3連複':'h','2連単':'c','2連複':'h'};
    for(const t of ['3連単','3連複','2連単','2連複']){
      const mk = t===r.recommend.type?' ◀ いちおし':'';
      h+=tipBlock(`【${t}】各3点${mk}`,cls[t],r.bet_types[t]);
    }
  }
  h+='<h3 class="h" style="opacity:.7">── 3連単を広げるなら ──</h3>';
  h+=tipBlock('◎ 本命（上位5点）','h',r.honmei);
  h+=tipBlock('○ 中穴（妙味5点）','c',r.chuuana);
  h+=tipBlock('★ 大穴（一発5点）','o',r.ooana);
  h+='</div>';
  // コメント
  let cmt='';
  if(r.kind.startsWith('波乱'))cmt=`⚠ 大穴注意レース！ 1号艇級の絶対本命が不在で波乱含み。大穴【${r.ooana[0].combo}】に妙味。`;
  else if(r.kind==='やや混戦')cmt=`やや混戦。中心はいるが崩れる余地あり。中穴まで広げる価値あり。`;
  else cmt=`堅い決着が見込めるレース。本命【${r.honmei[0].combo}】中心で。`;
  h+=`<div class="cmt">${cmt}</div>`;
  h+=`<div class="disc">⚠ 過去検証での回収率は最大93%（100%未満）。舟券は余裕資金の範囲で。</div>`;
  h+='</div>';
  $('result').innerHTML=h;
}
function tipBlock(title,cls,arr){
  let h=`<h3 class="${cls}">${title}</h3>`;
  for(const c of arr){
    const be = c.be_odds?` <span style="color:#7fdfff">${c.be_odds.toFixed(1)}倍〜買い</span>`:'';
    h+=`<div class="tip"><span>${c.combo}</span><span><b>${(c.prob*100).toFixed(1)}%</b> <span class="pay">約${c.payout.toLocaleString()}円</span>${be}</span></div>`;
  }
  return h;
}
boot();
</script>
</body></html>"""


def main():
    ip = local_ip()
    print("=" * 54)
    print(" 競艇AI予想  スマホ用サーバ起動")
    print("=" * 54)
    print(f" スマホのブラウザで次を開いてください（同じWi-Fi必須）:")
    print(f"   http://{ip}:{PORT}")
    print(f" PC自身で確認: http://localhost:{PORT}")
    print(" 終了する時はこの黒い画面を閉じてください。")
    print("=" * 54, flush=True)
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
