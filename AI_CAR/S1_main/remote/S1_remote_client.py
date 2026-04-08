# -*- coding: utf-8 -*-
"""
S1_remote_client.py - PC에서 실행
라즈베리 파이 자율주행차를 HTTP API로 완벽하게 제어하는 GUI.
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import io
import threading
import urllib.request
import urllib.error

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

DEFAULT_HOST = ''
DEFAULT_PORT = 5001
API_BASE = None


def api_url(path):
    if not API_BASE or not API_BASE[0]:
        raise ValueError('라즈베리 파이 IP를 입력하고 [연결]을 눌러주세요')
    return f'http://{API_BASE[0]}:{API_BASE[1]}{path}'


def api_get(path):
    try:
        req = urllib.request.Request(api_url(path), method='GET')
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {'ok': False, 'msg': str(e)}


def api_post(path, data=None):
    try:
        body = json.dumps(data or {}).encode() if data else b'{}'
        req = urllib.request.Request(
            api_url(path),
            data=body,
            method='POST',
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {'ok': False, 'msg': str(e)}


def api_get_frame():
    try:
        if not API_BASE or not API_BASE[0]:
            return None
        req = urllib.request.Request(api_url('/frame'), method='GET')
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.read()
    except Exception:
        return None


class S1RemoteClient:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('S1 자율주행차 원격 제어')
        self.root.geometry('640x820')
        self.root.resizable(True, True)

        self.host = tk.StringVar(value=DEFAULT_HOST)
        self.port = tk.IntVar(value=DEFAULT_PORT)
        self.connected = False
        self.log_messages = []
        self._last_process_output = []
        self._reconnect_fail_count = 0

        self.params = {
            'distance_threshold_cm': tk.IntVar(value=4),
            'distance_stop_sec': tk.IntVar(value=2),
            'speed': tk.IntVar(value=100),
            'servo_turn_angle': tk.IntVar(value=40),
            'red_light_stop_sec': tk.IntVar(value=2),
            'red_light_size_threshold': tk.IntVar(value=50),
            'start_flag': tk.BooleanVar(value=False),
            'camera_id': tk.IntVar(value=70),
            'epochs': tk.IntVar(value=10),
            'batch_size': tk.IntVar(value=32),
            'resume': tk.BooleanVar(value=False),
        }

        self._build_ui()
        self._start_polling()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill='both', expand=True)

        # ---- 연결 ----
        conn_frame = ttk.LabelFrame(main, text='라즈베리 파이 연결', padding=8)
        conn_frame.pack(fill='x', pady=4)
        row = ttk.Frame(conn_frame)
        row.pack(fill='x')
        ttk.Label(row, text='Pi IP 주소:').pack(side='left', padx=2)
        ttk.Entry(row, textvariable=self.host, width=18).pack(side='left', padx=2)
        ttk.Label(row, text='포트:').pack(side='left', padx=(8, 2))
        ttk.Entry(row, textvariable=self.port, width=6).pack(side='left', padx=2)
        ttk.Button(row, text='연결', command=self._connect).pack(side='left', padx=8)
        self.conn_label = ttk.Label(row, text='미연결', foreground='gray')
        self.conn_label.pack(side='left', padx=4)
        ttk.Label(conn_frame, text='(Pi에서 python S1_remote_server.py 실행 후 연결)', foreground='gray', font=('', 8)).pack(anchor='w')

        # ---- 제어 버튼 ----
        ctrl_frame = ttk.LabelFrame(main, text='원격 제어', padding=8)
        ctrl_frame.pack(fill='x', pady=4)
        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text='실행:', font=('', 9, 'bold')).pack(side='left', padx=2)
        for mode, label in [
            ('setup', 'Setup'),
            ('collect', '수집'),
            ('process', '처리'),
            ('train', '학습'),
            ('run', 'AI실행'),
        ]:
            ttk.Button(row1, text=label, command=lambda m=mode: self._start(m)).pack(side='left', padx=2)
        ttk.Button(row1, text='정지', command=self._stop).pack(side='left', padx=8)
        row2 = ttk.Frame(ctrl_frame)
        row2.pack(fill='x', pady=4)
        ttk.Button(row2, text='▶ AI 주행 시작', command=self._run_start).pack(side='left', padx=2)
        ttk.Button(row2, text='⏹ AI 주행 정지', command=self._run_stop).pack(side='left', padx=2)

        # ---- 파라미터 ----
        param_frame = ttk.LabelFrame(main, text='파라미터 (실시간 조정)', padding=8)
        param_frame.pack(fill='x', pady=4)
        rows = [
            ('거리 임계값(cm)', 'distance_threshold_cm', 1, 30),
            ('거리 정지시간(초)', 'distance_stop_sec', 1, 10),
            ('주행 속도', 'speed', 1, 100),
            ('서보 회전각', 'servo_turn_angle', 10, 90),
            ('빨간불 정지시간', 'red_light_stop_sec', 1, 10),
            ('카메라 ID', 'camera_id', 0, 99),
            ('학습 에포크', 'epochs', 1, 100),
            ('배치 크기', 'batch_size', 8, 128),
        ]
        for label, key, lo, hi in rows:
            r = ttk.Frame(param_frame)
            r.pack(fill='x', pady=2)
            ttk.Label(r, text=label, width=18, anchor='w').pack(side='left')
            var = self.params.get(key)
            if var is not None:
                tk.Spinbox(r, from_=lo, to=hi, textvariable=var, width=8).pack(side='left', padx=4)
        ttk.Checkbutton(param_frame, text='학습 재개 (--resume)', variable=self.params['resume']).pack(anchor='w', pady=2)
        btn_row = ttk.Frame(param_frame)
        btn_row.pack(fill='x', pady=4)
        ttk.Button(btn_row, text='파라미터 적용', command=self._apply_params).pack(side='left', padx=2)
        ttk.Button(btn_row, text='새로고침', command=self._refresh_params).pack(side='left', padx=2)

        # ---- 진행 상태 ----
        prog_frame = ttk.LabelFrame(main, text='진행 상태', padding=8)
        prog_frame.pack(fill='x', pady=4)
        self.prog_label = ttk.Label(prog_frame, text='대기 중')
        self.prog_label.pack(side='left', padx=4)
        self.prog_bar = ttk.Progressbar(prog_frame, length=250, mode='determinate')
        self.prog_bar.pack(side='left', padx=4, fill='x', expand=True)

        # ---- 출력 ----
        status_frame = ttk.LabelFrame(main, text='Pi 프로세스 출력', padding=8)
        status_frame.pack(fill='both', expand=True, pady=4)
        self.status_text = scrolledtext.ScrolledText(status_frame, height=14, wrap='word', font=('Consolas', 9))
        self.status_text.pack(fill='both', expand=True)

    def _set_api_base(self):
        global API_BASE
        API_BASE = (self.host.get().strip(), self.port.get())

    def _connect(self):
        if not self.host.get().strip():
            messagebox.showerror('오류', '라즈베리 파이 IP를 입력해주세요')
            return
        self._set_api_base()
        try:
            r = api_get('/status')
        except ValueError as e:
            self.conn_label.config(text='연결 실패', foreground='red')
            self._log(str(e))
            return
        if r.get('ok'):
            self.connected = True
            self.conn_label.config(text='연결됨', foreground='green')
            self._refresh_params()
            self._log('연결 성공')
        else:
            self.conn_label.config(text='연결 실패', foreground='red')
            self._log(f'연결 실패: {r.get("msg", "Unknown")}')

    def _log(self, msg):
        self.log_messages.append(msg)
        if len(self.log_messages) > 30:
            self.log_messages.pop(0)
        self._refresh_output_display()

    def _refresh_output_display(self):
        output = self._last_process_output
        parts = []
        if output:
            parts.append('\n'.join(output))
        if self.log_messages:
            parts.append('--- 로그 ---\n' + '\n'.join(self.log_messages[-15:]))
        self.status_text.delete('1.0', 'end')
        self.status_text.insert('end', '\n\n'.join(parts) if parts else '(대기 중)')
        self.status_text.see('end')

    def _start(self, mode):
        if not self.connected:
            self._set_api_base()
        if not self.host.get().strip():
            messagebox.showerror('오류', 'Pi IP를 입력하고 [연결]을 눌러주세요')
            return
        if mode == 'collect':
            try:
                r = api_get('/status')
                if not r.get('ok'):
                    messagebox.showerror('연결 실패', 'Pi 서버에 연결할 수 없습니다.\n\n1) Pi에서 S1_remote_server.py 실행\n2) IP 확인\n3) 같은 WiFi/네트워크')
                    return
            except Exception as e:
                messagebox.showerror('연결 실패', f'Pi 연결 실패: {e}')
                return
        data = {
            'camera_id': self.params['camera_id'].get(),
            'epochs': self.params['epochs'].get(),
            'batch_size': self.params['batch_size'].get(),
            'speed': self.params['speed'].get(),
            'resume': self.params['resume'].get(),
            **{k: v.get() for k, v in self.params.items() if k in (
                'distance_threshold_cm', 'distance_stop_sec', 'servo_turn_angle',
                'red_light_stop_sec', 'red_light_size_threshold'
            )},
        }
        if mode == 'run':
            data['start_flag'] = self.params['start_flag'].get()
        r = api_post(f'/start/{mode}', data)
        if r.get('ok'):
            self._last_process_output = []
            self.prog_bar['value'] = 0
            self._log(f'[{mode}] {r.get("msg", "시작")}')
            if mode == 'collect':
                self._open_collect_window()
        elif mode == 'collect':
            messagebox.showerror('수집 실패', f'{r.get("msg", "")}\n\nPi에서 서버가 실행 중인지 확인하세요.')
        else:
            self._log(f'[{mode}] 실패: {r.get("msg", "")}')
            messagebox.showerror('오류', r.get('msg', '실패'))

    def _stop(self):
        if not self.connected:
            self._set_api_base()
        r = api_post('/stop')
        self._log(f'정지: {r.get("msg", "")}')

    def _run_start(self):
        self.params['start_flag'].set(True)
        self._apply_params()
        self._log('AI 주행 시작 신호 전송')

    def _run_stop(self):
        self.params['start_flag'].set(False)
        self._apply_params()
        self._log('AI 주행 정지 신호 전송')

    def _apply_params(self):
        if not self.connected:
            self._set_api_base()
        data = {k: v.get() for k, v in self.params.items()}
        r = api_post('/params', data)
        if r.get('ok'):
            self._log('파라미터 적용됨')
        else:
            self._log(f'파라미터 실패: {r.get("msg", "")}')

    def _refresh_params(self):
        if not self.connected:
            self._set_api_base()
        r = api_get('/params')
        if r.get('ok') and 'params' in r:
            for k, v in r['params'].items():
                if k in self.params:
                    try:
                        self.params[k].set(v)
                    except Exception:
                        pass

    def _poll_status(self):
        if not API_BASE or not API_BASE[0]:
            return False
        try:
            r = api_get('/status')
            if r.get('ok'):
                self._reconnect_fail_count = 0
                self.connected = True
                self.conn_label.config(text='연결됨', foreground='green')
                mode = r.get('mode')
                running = r.get('running', False)
                live = r.get('live', {})
                epoch_cur = r.get('epoch_current', 0)
                epoch_tot = r.get('epoch_total', 0) or 10
                progress = r.get('progress_pct', 0)
                if running and mode == 'train' and epoch_tot > 0:
                    self.prog_label.config(text=f'Epoch {epoch_cur}/{epoch_tot} ({progress}%)')
                    self.prog_bar['value'] = progress
                elif running and mode:
                    self.prog_label.config(text=f'{mode} 실행 중...')
                    self.prog_bar['value'] = 0
                else:
                    self.prog_label.config(text='대기 중' if not running else f'{mode} 실행 중')
                    self.prog_bar['value'] = progress if running else 0
                output = r.get('process_output', [])
                if output or self._last_process_output:
                    self._last_process_output = output
                    self._refresh_output_display()
                s = f"모드: {mode or '-'} | 실행: {'예' if running else '아니오'}"
                if live.get('distance_cm') is not None:
                    s += f" | 거리: {live['distance_cm']:.1f}cm"
                if live.get('prediction'):
                    s += f" | 예측: {live['prediction']}"
                if running and mode == 'train' and epoch_tot > 0:
                    s += f" | Epoch {epoch_cur}/{epoch_tot}"
                self.root.title(f'S1 자율주행차 제어 - {s}')
                return running
        except Exception:
            self._handle_connection_lost()
        return False

    def _handle_connection_lost(self):
        self._reconnect_fail_count = getattr(self, '_reconnect_fail_count', 0) + 1
        if self._reconnect_fail_count >= 3 and self.connected:
            self.connected = False
            self.conn_label.config(text='연결 끊김', foreground='orange')
            self._log('연결이 끊어졌습니다. [연결]을 눌러 재연결하세요.')

    def _open_collect_window(self):
        if not HAS_PIL:
            messagebox.showwarning('PIL 필요', '수집 화면 표시: pip install Pillow')
            return
        win = tk.Toplevel(self.root)
        win.title('데이터 수집 - Pi 카메라')
        win.geometry('680x560')
        win.protocol('WM_DELETE_WINDOW', lambda: self._close_collect(win))

        img_frame = ttk.Frame(win, padding=4)
        img_frame.pack(fill='both', expand=True)
        self._collect_label = ttk.Label(img_frame, text='연결 중... (방향키: 1직진 2좌 3우)')
        self._collect_label.pack(expand=True)
        for w in (img_frame, self._collect_label):
            w.bind('<Button-1>', lambda e: win.focus_set())

        ctrl_frame = ttk.Frame(win, padding=8)
        ctrl_frame.pack(fill='x')
        ttk.Label(ctrl_frame, text='분류:', font=('', 10, 'bold')).pack(side='left', padx=4)
        for key, label in [('1', '↑ 직진'), ('2', '← 좌회전'), ('3', '→ 우회전')]:
            ttk.Button(ctrl_frame, text=label, width=12, command=lambda k=key: self._collect_key(k)).pack(side='left', padx=4)
        ttk.Button(ctrl_frame, text='종료', command=lambda: self._close_collect(win)).pack(side='left', padx=4)

        self._collect_win = win
        self._collect_running = True
        self._collect_fail_count = 0
        win.bind('<KeyPress-1>', lambda e: self._collect_key('1'))
        win.bind('<KeyPress-2>', lambda e: self._collect_key('2'))
        win.bind('<KeyPress-3>', lambda e: self._collect_key('3'))
        win.bind('<Up>', lambda e: self._collect_key('1'))
        win.bind('<Left>', lambda e: self._collect_key('2'))
        win.bind('<Right>', lambda e: self._collect_key('3'))
        win.focus_set()
        self._poll_collect_frame()

    def _collect_key(self, key):
        try:
            api_post('/collect/key', {'key': key})
        except ValueError:
            pass

    def _close_collect(self, win):
        self._collect_running = False
        try:
            api_post('/stop')
        except Exception:
            pass
        win.destroy()

    def _poll_collect_frame(self):
        if not getattr(self, '_collect_running', False) or not hasattr(self, '_collect_win'):
            return
        try:
            if self._collect_win.winfo_exists():
                data = api_get_frame()
                if data and len(data) > 100:
                    self._collect_fail_count = 0
                    img = Image.open(io.BytesIO(data))
                    img.thumbnail((640, 480))
                    photo = ImageTk.PhotoImage(img)
                    self._collect_label.config(image=photo, text='', foreground='black')
                    self._collect_label.image = photo
                else:
                    self._collect_fail_count += 1
                    if self._collect_fail_count > 40:
                        self._collect_label.config(
                            text=f'카메라 연결 대기 중...\n\nPi IP: {API_BASE[0] if API_BASE else ""}\n카메라 ID: {self.params["camera_id"].get()}',
                            foreground='orange', font=('', 10))
                try:
                    r = api_get('/status')
                    if r.get('ok') and r.get('collect_counts'):
                        cnt = r['collect_counts']
                        self._collect_win.title(f'수집 - 직진:{cnt[0]} 좌:{cnt[1]} 우:{cnt[2]}')
                except Exception:
                    pass
        except Exception:
            self._collect_fail_count += 1
        if getattr(self, '_collect_running', False):
            self.root.after(80, self._poll_collect_frame)

    def _start_polling(self):
        def poll():
            running = self._poll_status()
            interval = 500 if running else 2000
            self.root.after(interval, poll)
        self.root.after(500, poll)

    def run(self):
        self.root.mainloop()


def main():
    app = S1RemoteClient()
    app.run()


if __name__ == '__main__':
    main()
