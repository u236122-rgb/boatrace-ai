# -*- coding: utf-8 -*-
"""
tunnel.py  ―  競艇AI予想を「外（別ネットワーク）」に公開する一体型ランチャ
============================================================
ローカルWebサーバ(web.py)を同一プロセス内で起動し、Cloudflareトンネルで
インターネットに公開する。スマホ用の公開URLをこの画面に大きく表示する。

  起動:  python tunnel.py
  （4_スマホ外部公開.bat から呼ばれる）

ポート競合を避けるため空きポートを自動選択し、サーバとトンネルを1つの窓で
管理するので「サーバ窓だけ落ちて開けない」事故が起きない。
"""
import os
import re
import sys
import glob
import socket
import threading
import subprocess
from http.server import ThreadingHTTPServer

import web


def find_cloudflared():
    cands = [
        r"C:\Users\user\AppData\Local\Microsoft\WinGet\Packages"
        r"\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe",
    ]
    cands += glob.glob(os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Cloudflare.cloudflared*\cloudflared.exe"))
    cands += [os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\cloudflared.exe"),
              "cloudflared"]
    for c in cands:
        if c == "cloudflared" or os.path.exists(c):
            return c
    return None


def free_port(start=8000, span=30):
    for p in range(start, start + span):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", p))
            s.close()
            return p
        except OSError:
            s.close()
    return start


def kill_old_tunnels():
    """過去に起動して残っている cloudflared を止める（古いURL放置・二重起動を防ぐ）。"""
    try:
        subprocess.run(["taskkill", "/F", "/IM", "cloudflared.exe"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def main():
    if not web.PASSWORD:
        print("外部公開にはパスワードが必要です。")
        print("  環境変数 KYOTEI_WEB_PASSWORD を設定してから起動してください。")
        print("  例) set KYOTEI_WEB_PASSWORD=推測されにくい文字列")
        input("Enterで終了..."); return
    kill_old_tunnels()
    port = free_port(8000)
    web.PORT = port
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", port), web.Handler)
    except OSError as e:
        print(f"サーバを起動できません（ポート{port}）: {e}")
        input("Enterで終了..."); return
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[1/2] ローカルサーバ起動 OK : http://localhost:{port}")

    cf = find_cloudflared()
    if not cf:
        print("cloudflared が見つかりません。")
        print("  winget install Cloudflare.cloudflared を実行してください。")
        input("Enterで終了..."); return
    print("[2/2] Cloudflareトンネル接続中 ... (数秒〜十数秒)")

    login = f"ユーザー名 {web.USERNAME} / パスワード（環境変数で設定したもの）"

    proc = subprocess.Popen(
        [cf, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")

    url = None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m and not url:
                url = m.group(0)
                # 現在のURLをファイルにも保存（あとで見返せるように）
                try:
                    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "public_url.txt"), "w", encoding="utf-8") as f:
                        f.write(url + "\n")
                except Exception:
                    pass
                bar = "=" * 60
                print("\n" + bar)
                print("  ★ スマホでこのURLを開いてください（モバイル回線でOK）:")
                print("       " + url)
                print("  ★ ログイン: " + login)
                print("  ※ このURLは public_url.txt にも保存しました。")
                print("  ※ この画面は開いたまま。閉じると外部アクセスは止まります。")
                print("  ※ URLは起動するたびに変わります（最新のこのURLを使う）。")
                print(bar + "\n", flush=True)
                # スマホのカメラで読めるQRコードを表示（URL手入力が不要）
                try:
                    import segno
                    print("  ↓ スマホのカメラでこのQRを読み取るとすぐ開けます ↓\n")
                    segno.make(url, error="m").terminal(compact=True)
                    print("", flush=True)
                except Exception:
                    pass
            elif ("ERR" in line) or ("failed" in line.lower()) or ("error" in line.lower()):
                print("  cloudflared: " + line, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        srv.shutdown()


if __name__ == "__main__":
    main()
