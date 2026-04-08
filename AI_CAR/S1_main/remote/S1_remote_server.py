# -*- coding: utf-8 -*-
"""
S1_remote_server.py - 라즈베리 파이에서 실행
PC에서 HTTP API로 자율주행차를 완벽하게 제어.
"""
import os
import sys
import re
import json
import subprocess
import threading
import socket

REMOTE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REMOTE_DIR)


def _get_kitinfo_path():
    local = os.path.join(REMOTE_DIR, 'kitinfo.json')
    if os.path.isfile(local):
        return local
    if sys.platform != 'win32' and os.path.isfile('/home/pi/wifi-scan/kitinfo.json'):
        return '/home/pi/wifi-scan/kitinfo.json'
    return local


PARAMS_FILE = os.path.join(REMOTE_DIR, 'remote_params.json')
LIVE_FILE = os.path.join(REMOTE_DIR, 'remote_live.json')
DEFAULT_PARAMS = {
    'distance_threshold_cm': 4,
    'distance_stop_sec': 2,
    'speed': 100,
    'servo_turn_angle': 40,
    'red_light_stop_sec': 2,
    'red_light_size_threshold': 50,
    'start_flag': False,
    'camera_id': 70,
    'epochs': 10,
    'batch_size': 32,
    'block_dir': None,
    'resume': False,
}

current_process = None
process_lock = threading.Lock()
current_mode = None
process_output_lines = []
process_output_lock = threading.Lock()
process_epoch_current = 0
process_epoch_total = 0
process_progress_pct = 0
MAX_OUTPUT_LINES = 200

collect_thread = None
collect_stop = threading.Event()
collect_frame = None
collect_frame_lock = threading.Lock()
collect_key_queue = []
collect_key_lock = threading.Lock()
collect_counts = [0, 0, 0]


def save_params(params):
    try:
        with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print('[remote] save_params 실패:', e)


def load_params():
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    save_params(DEFAULT_PARAMS.copy())
    return DEFAULT_PARAMS.copy()


def get_block_dir():
    bd = os.path.join(REMOTE_DIR, 'block')
    if os.path.isdir(bd) and os.path.isfile(os.path.join(bd, 'tflite_detectorlib.py')):
        return os.path.abspath(bd)
    return None


def _read_process_output(proc, mode):
    global process_output_lines, process_epoch_current, process_epoch_total, process_progress_pct
    epoch_re = re.compile(r'Epoch\s+(\d+)/(\d+)', re.I)
    step_re = re.compile(r'(\d+)/(\d+)\s+\[')
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            line = line.rstrip()
            if line:
                with process_output_lock:
                    process_output_lines.append(line)
                    if len(process_output_lines) > MAX_OUTPUT_LINES:
                        process_output_lines.pop(0)
                m = epoch_re.search(line)
                if not m:
                    m = step_re.search(line)
                if m:
                    cur, tot = int(m.group(1)), int(m.group(2))
                    process_epoch_current = cur
                    process_epoch_total = tot
                    process_progress_pct = int(100 * cur / tot) if tot > 0 else 0
    except Exception:
        pass


def run_s1_mode(mode, extra_args=None):
    global current_process, current_mode, process_output_lines, process_epoch_current, process_epoch_total, process_progress_pct
    with process_lock:
        if current_process and current_process.poll() is None:
            return {'ok': False, 'msg': f'이미 {current_mode} 실행 중'}
        with process_output_lock:
            process_output_lines = []
        process_epoch_current = 0
        process_epoch_total = 0
        process_progress_pct = 0
        args = [sys.executable, os.path.join(REMOTE_DIR, 'S1_main.py'), mode]
        if extra_args:
            args.extend(extra_args)
        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            try:
                current_process = subprocess.Popen(
                    args, cwd=REMOTE_DIR,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace',
                    bufsize=1, env=env,
                )
            except TypeError:
                current_process = subprocess.Popen(
                    args, cwd=REMOTE_DIR,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, env=env,
                )
            current_mode = mode
            t = threading.Thread(target=_read_process_output, args=(current_process, mode), daemon=True)
            t.start()
            return {'ok': True, 'msg': f'{mode} 시작됨'}
        except Exception as e:
            return {'ok': False, 'msg': str(e)}


def _remote_collect_loop(camera_id):
    global collect_frame, collect_counts, collect_thread
    import time
    import cv2
    motor = servo = None
    kit_middle = 90
    turn_angle = 40
    motor_direction_back = 'N'
    if sys.platform != 'win32':
        kit_path = _get_kitinfo_path()
        if os.path.isfile(kit_path):
            try:
                with open(kit_path, 'r') as f:
                    info = json.load(f)
                kit_middle = int(info.get('servo_angle', 90))
                motor_direction_back = info.get('motor_direction_back', 'N')
            except Exception:
                pass
        try:
            from gpiozero import Motor, AngularServo
            from gpiozero.pins.pigpio import PiGPIOFactory
            factory = PiGPIOFactory()
            motor = Motor(forward=19, backward=13, pin_factory=factory) if motor_direction_back == 'N' else Motor(forward=13, backward=19, pin_factory=factory)
            servo = AngularServo(12, pin_factory=factory)
            servo.angle = kit_middle
            print('[수집] 모터/서보 활성화')
        except Exception:
            pass
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 15)
        names_path = os.path.join(REMOTE_DIR, 'names.json')
        if not os.path.isfile(names_path):
            class_list = ['직진', '좌회전', '우회전']
            with open(names_path, 'w', encoding='utf-8') as f:
                json.dump({'names': class_list}, f, ensure_ascii=False, indent=2)
        else:
            with open(names_path, 'r', encoding='utf-8') as f:
                class_list = json.load(f)['names']
        n_classes = len(class_list)
        while len(collect_counts) < n_classes:
            collect_counts.append(0)
        for i in range(n_classes):
            class_dir = os.path.join(REMOTE_DIR, 'data', 'training', class_list[i])
            os.makedirs(class_dir, exist_ok=True)
            os.makedirs(os.path.join(class_dir, 'thumb'), exist_ok=True)
        while not collect_stop.is_set():
            params = load_params()
            turn_angle = params.get('servo_turn_angle', 40)
            speed = params.get('speed', 100) / 100.0
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, -1)
            disp = frame.copy()
            cv2.putText(disp, '1:직진 2:좌 3:우 (PC에서 조종)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for i, name in enumerate(class_list):
                cv2.putText(disp, f'{name}: {collect_counts[i]}', (10, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            with collect_frame_lock:
                _, jpg = cv2.imencode('.jpg', disp)
                collect_frame = jpg.tobytes()
            with collect_key_lock:
                keys = list(collect_key_queue)
                collect_key_queue.clear()
            for key in keys:
                if key in ('1', '2', '3'):
                    idx = int(key) - 1
                    class_name = class_list[idx]
                    class_dir = os.path.join(REMOTE_DIR, 'data', 'training', class_name)
                    fname = f'{idx}_{int(time.time()*1000)}_{collect_counts[idx]}.jpg'
                    path = os.path.join(class_dir, fname)
                    cv2.imwrite(path, frame)
                    thumb = cv2.resize(frame, (90, 70), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(class_dir, 'thumb', f'thumb_{fname}'), thumb)
                    collect_counts[idx] += 1
                    if motor is not None and servo is not None:
                        motor.forward(speed=speed)
                        if key == '1':
                            servo.angle = kit_middle
                        elif key == '2':
                            servo.angle = kit_middle + turn_angle
                        elif key == '3':
                            servo.angle = kit_middle - turn_angle
            time.sleep(0.05)
        cap.release()
    except Exception as e:
        print('collect error:', e)
    finally:
        if motor is not None:
            motor.stop()
        if servo is not None:
            servo.angle = kit_middle
        collect_thread = None


def start_remote_collect(camera_id=70):
    global collect_thread, collect_stop, collect_counts, current_mode
    with process_lock:
        if current_process and current_process.poll() is None:
            return {'ok': False, 'msg': f'이미 {current_mode} 실행 중'}
        if collect_thread and collect_thread.is_alive():
            return {'ok': False, 'msg': '이미 수집 실행 중'}
    collect_stop.clear()
    collect_counts[:] = [0, 0, 0]
    current_mode = 'collect'
    collect_thread = threading.Thread(target=_remote_collect_loop, args=(camera_id,))
    collect_thread.start()
    return {'ok': True, 'msg': '원격 수집 시작'}


def stop_remote_collect():
    global current_mode
    collect_stop.set()
    current_mode = None
    return {'ok': True, 'msg': '수집 정지'}


def collect_send_key(key):
    with collect_key_lock:
        collect_key_queue.append(key)
    return {'ok': True}


def stop_current():
    global current_process, current_mode
    with process_lock:
        if current_process and current_process.poll() is None:
            current_process.terminate()
            try:
                current_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                current_process.kill()
            current_process = None
            current_mode = None
            return {'ok': True, 'msg': '정지됨'}
        return {'ok': True, 'msg': '실행 중인 프로세스 없음'}


try:
    from flask import Flask, request, jsonify, Response
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

if HAS_FLASK:
    app = Flask(__name__)
    try:
        from flask_cors import CORS
        CORS(app)
    except ImportError:
        pass

    @app.route('/status', methods=['GET'])
    def api_status():
        with process_lock:
            proc_running = current_process is not None and current_process.poll() is None
        collect_running = collect_thread is not None and collect_thread.is_alive()
        running = proc_running or collect_running
        params = load_params()
        live = {}
        if os.path.isfile(LIVE_FILE):
            try:
                with open(LIVE_FILE, 'r') as f:
                    live = json.load(f)
            except Exception:
                pass
        with process_output_lock:
            output = list(process_output_lines)
        return jsonify({
            'ok': True, 'running': running,
            'mode': 'collect' if collect_running else current_mode,
            'collect_counts': collect_counts if collect_running else None,
            'params': params, 'live': live,
            'process_output': output,
            'epoch_current': process_epoch_current,
            'epoch_total': process_epoch_total if process_epoch_total > 0 else params.get('epochs', 10),
            'progress_pct': process_progress_pct,
        })

    @app.route('/frame', methods=['GET'])
    def api_frame():
        with collect_frame_lock:
            frame = collect_frame
        if frame is None:
            return Response(b'', status=404, mimetype='image/jpeg')
        return Response(frame, mimetype='image/jpeg')

    @app.route('/collect/key', methods=['POST'])
    def api_collect_key():
        data = request.get_json() or {}
        key = str(data.get('key', ''))
        if key not in ('1', '2', '3'):
            return jsonify({'ok': False, 'msg': 'key는 1,2,3'}), 400
        return jsonify(collect_send_key(key))

    @app.route('/start/<mode>', methods=['POST'])
    def api_start(mode):
        if mode not in ('setup', 'collect', 'process', 'train', 'run', 'full'):
            return jsonify({'ok': False, 'msg': f'잘못된 mode: {mode}'}), 400
        data = request.get_json() or {}
        if mode == 'collect':
            cam_id = data.get('camera_id', data.get('camera', 70))
            return jsonify(start_remote_collect(cam_id))
        extra = []
        cam = data.get('camera', data.get('camera_id', 70))
        if 'camera' in data or 'camera_id' in data:
            extra.extend(['--camera', str(cam)])
        if 'epochs' in data:
            extra.extend(['--epochs', str(data['epochs'])])
        if 'batch' in data:
            extra.extend(['--batch', str(data['batch'])])
        if data.get('resume'):
            extra.append('--resume')
        if 'speed' in data:
            extra.extend(['--speed', str(data['speed'])])
        if 'block_dir' in data and data['block_dir']:
            extra.extend(['--block-dir', str(data['block_dir'])])
        elif mode == 'run':
            bd = data.get('block_dir') or get_block_dir()
            if bd:
                extra.extend(['--block-dir', str(bd)])
        params_update = {k: v for k, v in data.items() if k in DEFAULT_PARAMS}
        if mode == 'run':
            params_update['start_flag'] = data.get('start_flag', False)
            extra.extend(['--params-file', PARAMS_FILE])
        save_params({**load_params(), **params_update})
        result = run_s1_mode(mode, extra if extra else None)
        return jsonify(result)

    @app.route('/stop', methods=['POST'])
    def api_stop():
        if collect_thread and collect_thread.is_alive():
            return jsonify(stop_remote_collect())
        return jsonify(stop_current())

    @app.route('/params', methods=['GET', 'POST'])
    def api_params():
        if request.method == 'GET':
            return jsonify({'ok': True, 'params': load_params()})
        data = request.get_json()
        if not data:
            return jsonify({'ok': False, 'msg': 'JSON 필요'}), 400
        params = load_params()
        for k in DEFAULT_PARAMS:
            if k in data:
                params[k] = data[k]
        save_params(params)
        return jsonify({'ok': True, 'params': params})


def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass
    try:
        import netifaces
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:
                for a in addrs[netifaces.AF_INET]:
                    ip = a.get('addr', '')
                    if ip and not ip.startswith('127.'):
                        return ip
    except ImportError:
        pass
    fallback = socket.gethostbyname(socket.gethostname())
    if fallback and fallback != '127.0.0.1':
        return fallback
    return '127.0.0.1'


def main():
    import argparse
    load_params()
    if not HAS_FLASK:
        print('pip install flask flask-cors 필요')
        sys.exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=5001)
    args = parser.parse_args()
    port = args.port
    host = '0.0.0.0'
    lan_ip = get_lan_ip()
    print('=' * 50)
    print(f'S1 자율주행 원격 서버: http://{lan_ip}:{port}')
    print('PC에서 이 주소로 연결하여 제어하세요.')
    print('Ctrl+C 종료')
    print('=' * 50)
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
