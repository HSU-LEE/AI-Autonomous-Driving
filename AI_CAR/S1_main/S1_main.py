# -*- coding: utf-8 -*-
"""
S1_main.py - Copy_도로_분류 프로젝트 참고 (독립 실행용)
데이터 수집 → 데이터 분류/처리 → 모델 설계 → 에포크 학습 → AI 실행

어느 폴더에 두고 실행해도, 스크립트 위치를 기준으로 data/model 등 생성됨.
"""
import os
import sys
import random
import time
import argparse

# 단독 실행: 스크립트가 있는 폴더 = 프로젝트 루트 (S1 폴더만 복사해도 동작)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = SCRIPT_DIR


def _get_kitinfo_path():
    """kitinfo.json 경로 (S1 폴더 > wifi-scan 키트 설정)"""
    local = os.path.join(PROJECT_PATH, 'kitinfo.json')
    if os.path.isfile(local):
        return local
    # 키트 기본 경로 (서보/모터 보정값)
    if sys.platform != 'win32' and os.path.isfile('/home/pi/wifi-scan/kitinfo.json'):
        return '/home/pi/wifi-scan/kitinfo.json'
    return local

# Copy_도로_분류와 동일한 클래스
CLASS_NAMES = ['직진', '좌회전', '우회전']


# ==================== 1. 데이터 수집 ====================
def setup_data_collection():
    """클래스 폴더 생성 및 names.json 초기화"""
    os.makedirs(PROJECT_PATH, exist_ok=True)
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(PROJECT_PATH, 'data', 'training', class_name)
        os.makedirs(class_dir, exist_ok=True)
        os.makedirs(os.path.join(class_dir, 'thumb'), exist_ok=True)
    
    import json
    names_path = os.path.join(PROJECT_PATH, 'names.json')
    with open(names_path, 'w', encoding='utf-8') as f:
        json.dump({'names': CLASS_NAMES}, f, ensure_ascii=False, indent=2)
    print(f'[데이터 수집] 클래스 폴더 생성 완료: {CLASS_NAMES}')


def collect_data(camera_id=70):
    """키보드로 데이터 수집 (1:직진, 2:좌회전, 3:우회전, q:종료)"""
    import cv2
    import json
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    with open(os.path.join(PROJECT_PATH, 'names.json'), 'r', encoding='utf-8') as f:
        class_list = json.load(f)['names']
    
    print('=== 데이터 수집 모드 ===')
    print('1: 직진 | 2: 좌회전 | 3: 우회전 | q: 종료')
    
    cnt = {i: 0 for i in range(len(class_list))}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, -1)
        disp = frame.copy()
        cv2.putText(disp, '1:직진 2:좌 3:우 q:quit', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for i, name in enumerate(class_list):
            cv2.putText(disp, f'{name}: {cnt[i]}', (10, 60 + i * 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Data Collection', disp)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        if key in (ord('1'), ord('2'), ord('3')):
            idx = key - ord('1')
            class_name = class_list[idx]
            class_dir = os.path.join(PROJECT_PATH, 'data', 'training', class_name)
            fname = f'{idx}_{int(time.time()*1000)}_{cnt[idx]}.jpg'
            path = os.path.join(class_dir, fname)
            cv2.imwrite(path, frame)
            thumb_path = os.path.join(class_dir, 'thumb', f'thumb_{fname}')
            thumb = cv2.resize(frame, (90, 70), interpolation=cv2.INTER_AREA)
            cv2.imwrite(thumb_path, thumb)
            cnt[idx] += 1
            print(f'  저장: {class_name} ({cnt[idx]}장)')
    
    cap.release()
    cv2.destroyAllWindows()
    print(f'총 수집: {sum(cnt.values())}장')


# ==================== 2. 데이터 분류 및 처리 ====================
def create_csv(csv_name='훈련데이터.csv'):
    """훈련 데이터 CSV 생성"""
    import csv
    import json
    
    with open(os.path.join(PROJECT_PATH, 'names.json'), 'r', encoding='utf-8') as f:
        classes = json.load(f)['names']
    
    csv_path = os.path.join(PROJECT_PATH, csv_name)
    total = 0
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(PROJECT_PATH, 'data', 'training', class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith(('.jpg', '.jpeg', '.png')) and not fname.startswith('thumb'):
                    img_path = os.path.join(class_dir, fname)
                    if os.path.isfile(img_path):
                        rel_path = os.path.join('data', 'training', class_name, fname).replace('\\', '/')
                        writer.writerow((rel_path, idx))
                        total += 1
    print(f'[데이터 처리] CSV 생성: {csv_name} ({total}장)')


def decalcom_augment():
    """데이터 증강: 좌우 반전"""
    import json
    import cv2
    
    with open(os.path.join(PROJECT_PATH, 'names.json'), 'r', encoding='utf-8') as f:
        classes = json.load(f)['names']
    
    augment_map = [('직진', '직진'), ('좌회전', '우회전'), ('우회전', '좌회전')]
    
    for from_class, to_class in augment_map:
        from_dir = os.path.join(PROJECT_PATH, 'data', 'training', from_class)
        to_dir = os.path.join(PROJECT_PATH, 'data', 'training', to_class)
        if not os.path.isdir(from_dir):
            continue
        os.makedirs(os.path.join(to_dir, 'thumb'), exist_ok=True)
        
        for fname in os.listdir(from_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')) or fname.startswith('thumb'):
                continue
            if fname.startswith('dc_'):
                continue
            img_path = os.path.join(from_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            flipped = cv2.flip(img, 1)
            dc_name = f'dc_{fname}'
            dc_path = os.path.join(to_dir, dc_name)
            if os.path.exists(dc_path):
                continue
            cv2.imwrite(dc_path, flipped)
            thumb = cv2.resize(flipped, (90, 70), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(to_dir, 'thumb', f'thumb_{dc_name}'), thumb)
    print('[데이터 처리] 증강(데칼코마니) 완료')


def analysis_csv(csv_name='훈련데이터.csv'):
    """CSV 데이터 분석"""
    import csv
    import json
    
    with open(os.path.join(PROJECT_PATH, 'names.json'), 'r', encoding='utf-8') as f:
        classes = json.load(f)['names']
    
    csv_path = os.path.join(PROJECT_PATH, csv_name)
    if not os.path.exists(csv_path):
        print(f'CSV 없음: {csv_name}')
        return
    
    counts = [0] * len(classes)
    with open(csv_path, 'r', newline='', encoding='utf-8') as cf:
        for row in csv.reader(cf):
            if len(row) >= 2:
                try:
                    idx = int(row[1])
                    if 0 <= idx < len(classes):
                        counts[idx] += 1
                except (ValueError, IndexError):
                    pass
    
    total = sum(counts)
    if total == 0:
        print('데이터 없음')
        return
    print(f'[데이터 분석] {csv_name}')
    print(f'  총 {total}장')
    for i, c in enumerate(classes):
        print(f'  {c}: {counts[i]} ({100*counts[i]/total:.1f}%)')


def upsampling_csv(csv_name='훈련데이터.csv'):
    """클래스 불균형 업샘플링"""
    import csv
    import json
    
    with open(os.path.join(PROJECT_PATH, 'names.json'), 'r', encoding='utf-8') as f:
        classes = json.load(f)['names']
    
    csv_path = os.path.join(PROJECT_PATH, csv_name)
    rows_by_class = [[] for _ in classes]
    
    with open(csv_path, 'r', newline='', encoding='utf-8') as cf:
        for row in csv.reader(cf):
            if len(row) >= 2:
                try:
                    idx = int(row[1])
                    if 0 <= idx < len(rows_by_class):
                        rows_by_class[idx].append(row)
                except (ValueError, IndexError):
                    pass
    
    non_empty = [r for r in rows_by_class if len(r) > 0]
    if not non_empty:
        print('[데이터 처리] 업샘플링 불가: 데이터 없음')
        return
    max_count = max(len(r) for r in rows_by_class)
    to_add = []
    for i in range(len(classes)):
        if len(rows_by_class[i]) == 0:
            print(f'  경고: {classes[i]} 클래스에 이미지 없음')
            continue
        while len(rows_by_class[i]) < max_count:
            sample = random.choice(rows_by_class[i])
            rows_by_class[i].append(sample)
            to_add.append(sample)
    
    if to_add:
        with open(csv_path, 'a', newline='', encoding='utf-8') as cf:
            csv.writer(cf).writerows(to_add)
        print(f'[데이터 처리] 업샘플링: {len(to_add)}행 추가')
    else:
        print('[데이터 처리] 업샘플링 불필요')


# ==================== 3. 모델 설계 ====================
def build_model(num_classes=3):
    """32x32 CNN 모델"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, Dense
    
    model = Sequential([
        InputLayer(input_shape=(32, 32, 3)),
        Conv2D(6, (5, 5), (1, 1), activation='relu'),
        MaxPool2D((2, 2), (2, 2)),
        Conv2D(16, (5, 5), (1, 1), activation='relu'),
        MaxPool2D((2, 2), (2, 2)),
        Conv2D(120, (5, 5), (1, 1), activation='relu'),
        Flatten(),
        Dense(84, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    return model


# ==================== 4. 에포크 학습 ====================
def train_model(epochs=10, batch_size=32, csv_name='훈련데이터.csv', first_training=True):
    """CSV 기반 훈련"""
    import csv
    import numpy as np
    import cv2
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.utils import Sequence
    
    csv_path = os.path.join(PROJECT_PATH, csv_name)
    if not os.path.exists(csv_path):
        print(f'CSV 없음: {csv_name}. 먼저 Setup → 수집 → 처리 실행')
        return
    
    import json
    with open(os.path.join(PROJECT_PATH, 'names.json'), 'r', encoding='utf-8') as f:
        classes = json.load(f)['names']
    num_classes = len(classes)
    
    xs, ys = [], []
    with open(csv_path, 'r', newline='', encoding='utf-8') as cf:
        for row in csv.reader(cf):
            if len(row) >= 2:
                try:
                    y = int(row[1])
                    if y < 0 or y >= num_classes:
                        continue
                except ValueError:
                    continue
                rel_path = row[0].strip()
                full_path = os.path.join(PROJECT_PATH, rel_path) if not os.path.isabs(rel_path) else rel_path
                if os.path.exists(full_path):
                    xs.append(full_path)
                    ys.append(y)
    
    if len(xs) == 0:
        print('학습 데이터 없음')
        return
    
    if max(ys) >= num_classes or min(ys) < 0:
        print('레이블 오류')
        return
    
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)
    split = int(len(xs) * 0.8)
    train_xs, train_ys = list(xs[:split]), list(ys[:split])
    val_xs, val_ys = list(xs[split:]), list(ys[split:])
    
    _load_warned = set()
    def load_train_batch(batch_size, idx):
        x_out, y_out = [], []
        for i in range(batch_size):
            j = (idx * batch_size + i) % len(train_xs)
            img = cv2.imread(train_xs[j])
            if img is None:
                if train_xs[j] not in _load_warned:
                    _load_warned.add(train_xs[j])
                    print(f'  [경고] 이미지 로드 실패 (검은 화면 사용): {train_xs[j]}')
                img = np.zeros((32, 32, 3), dtype=np.uint8)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            x_out.append(img)
            y_out.append(train_ys[j])
        return np.array(x_out), np.array(y_out)
    
    def load_val_batch(batch_size, idx):
        x_out, y_out = [], []
        for i in range(batch_size):
            j = (idx * batch_size + i) % len(val_xs)
            img = cv2.imread(val_xs[j])
            if img is None:
                if val_xs[j] not in _load_warned:
                    _load_warned.add(val_xs[j])
                    print(f'  [경고] 이미지 로드 실패 (검은 화면 사용): {val_xs[j]}')
                img = np.zeros((32, 32, 3), dtype=np.uint8)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            x_out.append(img)
            y_out.append(val_ys[j])
        return np.array(x_out), np.array(y_out)
    
    class TrainSequence(Sequence):
        def __init__(self, total, batch_size):
            self.total = total
            self.batch_size = batch_size
        def __len__(self):
            return int(np.ceil(self.total / self.batch_size))
        def __getitem__(self, idx):
            x, y = load_train_batch(self.batch_size, idx)
            return x, to_categorical(y, num_classes=num_classes)
    
    class ValSequence(Sequence):
        def __init__(self, total, batch_size):
            self.total = total
            self.batch_size = batch_size
        def __len__(self):
            return int(np.ceil(self.total / self.batch_size))
        def __getitem__(self, idx):
            x, y = load_val_batch(self.batch_size, idx)
            return x, to_categorical(y, num_classes=num_classes)
    
    model = build_model(num_classes=num_classes)
    model_dir = os.path.join(PROJECT_PATH, 'model')
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, 'model.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path, save_weights_only=True, verbose=0
    )

    class EpochProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f'\n--- Epoch {epoch + 1}/{epochs} ---')

        def on_epoch_end(self, epoch, logs=None):
            if logs:
                loss = logs.get('loss', 0)
                acc = logs.get('accuracy', 0) * 100
                val_loss = logs.get('val_loss', 0)
                val_acc = logs.get('val_accuracy', 0) * 100
                print(f'  loss: {loss:.4f} | acc: {acc:.2f}% | val_loss: {val_loss:.4f} | val_acc: {val_acc:.2f}%')
            print(f'  진행률: {(epoch + 1) / epochs * 100:.0f}% (체크포인트 저장됨)')
    
    if not first_training and os.path.exists(ckpt_path + '.index'):
        model.load_weights(ckpt_path)
        print('[학습] 이전 모델 가중치 로드')
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    
    train_seq = TrainSequence(len(train_xs), batch_size)
    val_seq = ValSequence(len(val_xs), batch_size)
    
    print('=' * 60)
    print(f'[학습 시작] epochs={epochs} | batch_size={batch_size}')
    print(f'  훈련 데이터: {len(train_xs)}장 | 검증 데이터: {len(val_xs)}장')
    print('=' * 60)
    callbacks = [cp_callback, EpochProgressCallback()]
    model.fit(train_seq, validation_data=val_seq, epochs=epochs, callbacks=callbacks, verbose=0)
    model.save_weights(ckpt_path)
    print('=' * 60)
    print(f'[학습 완료] {ckpt_path}')
    print('=' * 60)


# ==================== 5. AI 실행 ====================
def run_ai(camera_id=70, speed=100, block_dir=None, params_file=None):
    """학습된 모델로 도로 분류 AI 실행"""
    import cv2
    import numpy as np
    import json

    # S1_main 내장 block: block_dir를 sys.path에 추가
    if block_dir and os.path.isdir(block_dir) and block_dir not in sys.path:
        sys.path.insert(0, block_dir)

    tof = None
    try:
        if sys.platform != 'win32' and block_dir:
            import VL53L0X
            tof = VL53L0X.VL53L0X()
            tof.start_ranging(VL53L0X.VL53L0X_BETTER_ACCURACY_MODE)
            print('[AI] 거리센서 활성화')
    except Exception:
        pass

    kit_middle = 90
    adjust_angle = 0
    motor_direction_back = 'N'
    kit_path = _get_kitinfo_path()
    if os.path.isfile(kit_path):
        try:
            with open(kit_path, 'r') as f:
                info = json.load(f)
            kit_middle = int(info.get('servo_angle', 90))
            adjust_angle = int(info.get('adjust_angle', 0))
            motor_direction_back = info.get('motor_direction_back', 'N')
        except Exception:
            pass

    motor = servo = None
    use_hw = False
    try:
        if sys.platform != 'win32':
            from gpiozero import Motor, AngularServo
            from gpiozero.pins.pigpio import PiGPIOFactory
            factory = PiGPIOFactory()
            motor = Motor(forward=19, backward=13, pin_factory=factory) if motor_direction_back == 'N' else Motor(forward=13, backward=19, pin_factory=factory)
            servo = AngularServo(12, pin_factory=factory)
            servo.angle = kit_middle
            use_hw = True
            print('[AI] 모터/서보 활성화')
    except Exception:
        pass

    tf_detector = None
    tf_check = None
    if block_dir and os.path.isdir(block_dir):
        try:
            if block_dir not in sys.path:
                sys.path.insert(0, block_dir)
            import tflite_detectorlib as _tf
            import traffic_color_check as _tc
            tf_detector = _tf
            tf_check = _tc
            print('[AI] 신호등 탐지 활성화')
        except Exception:
            pass

    def traffic_sign_detector(cam_image):
        if tf_detector is None:
            return (None, None)
        try:
            rtn = tf_detector.detect1(cam_image)
            return (rtn[2], rtn[1])
        except Exception:
            return (None, None)

    with open(os.path.join(PROJECT_PATH, 'names.json'), 'r', encoding='utf-8') as f:
        classes = json.load(f)['names']
    model = build_model(num_classes=len(classes))
    ckpt_path = os.path.join(PROJECT_PATH, 'model', 'model.ckpt')
    if not (os.path.exists(ckpt_path + '.index') or os.path.isfile(ckpt_path) or os.path.isdir(ckpt_path)):
        print('학습된 모델 없음. train_model() 먼저 실행')
        return
    model.load_weights(ckpt_path)

    cam = None
    cap = None
    if block_dir and os.path.isdir(block_dir):
        try:
            if block_dir not in sys.path:
                sys.path.insert(0, block_dir)
            import cameralib as camera
            cam = camera.Camera(camera_id=camera_id)
            cam.initialize()
            print('[AI] cameralib 카메라 사용')
        except Exception:
            pass
    if cam is None:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 15)
        print('[AI] cv2 카메라 사용')

    def get_frame():
        if cam is not None:
            return cam.get_frame()
        ret, f = cap.read()
        return cv2.flip(f, -1) if ret else None

    def load_params():
        if not params_file or not os.path.isfile(params_file):
            return {}
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def write_live_data(dist_cm=None, pred=None, light=None):
        if not params_file:
            return
        live_path = os.path.join(os.path.dirname(params_file), 'remote_live.json')
        try:
            data = {}
            if os.path.isfile(live_path):
                try:
                    with open(live_path, 'r') as f:
                        data = json.load(f)
                except Exception:
                    pass
            if dist_cm is not None:
                data['distance_cm'] = dist_cm
            if pred is not None:
                data['prediction'] = pred
            if light is not None:
                data['traffic_light'] = light
            with open(live_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass

    # params_file 있으면 PC 원격 제어, 없으면 로컬 키보드 제어
    last_direction = None
    start_flag = False
    if params_file:
        print('=' * 50)
        print('=== AI 자율주행 (PC 원격 제어) ===')
        print('  PC에서 시작/정지 버튼으로 제어')
        print('=' * 50)
    else:
        print('=' * 50)
        print('=== AI 자율주행 (로컬 키보드) ===')
        print('  a: 주행 시작 | h: 주행 정지 | q: 종료')
        print('=' * 50)

    try:
        while True:
            dist_cm = None
            rp = load_params()
            dist_thresh = rp.get('distance_threshold_cm', 4)
            dist_stop_sec = rp.get('distance_stop_sec', 2)
            run_speed = rp.get('speed', speed) if rp else speed
            turn_angle = rp.get('servo_turn_angle', 40)
            red_light_stop_sec = rp.get('red_light_stop_sec', 2)
            if params_file:
                start_flag = rp.get('start_flag', False)

            if tof is not None and use_hw:
                dist_cm = float(tof.get_distance() / 10)
                write_live_data(dist_cm=dist_cm)
                if dist_cm <= dist_thresh:
                    motor.stop()
                    if last_direction is None:
                        servo.angle = kit_middle
                    elif last_direction == 'Left':
                        servo.angle = kit_middle + adjust_angle
                    elif last_direction == 'Right':
                        servo.angle = kit_middle - adjust_angle
                    time.sleep(dist_stop_sec)
                    continue

            cam_image = get_frame()
            if cam_image is None:
                continue

            traffic_light_color = None
            try:
                traffic_light, traffic_speed = traffic_sign_detector(cam_image)
                if traffic_light is not None and len(traffic_light) >= 5 and traffic_light[0] is not None:
                    if traffic_light[0] == 'red':
                        if tf_check is not None:
                            tf_check.red_append(traffic_light[4])
                        traffic_light_color = 'red'
                    else:
                        if tf_check is not None:
                            tf_check.red_append(0)
                        traffic_light_color = traffic_light[0]
                else:
                    if tf_check is not None:
                        tf_check.red_append(0)
                        if tf_check.max_size() > 50:
                            continue
                    traffic_light_color = None
            except Exception:
                pass

            if traffic_light_color == 'red':
                if use_hw:
                    motor.stop()
                time.sleep(red_light_stop_sec)
                continue

            try:
                h_img = cam_image.shape[0]
                road_image = cam_image[230:] if h_img > 230 else cam_image
                road_image = cv2.resize(road_image, (32, 32), interpolation=cv2.INTER_AREA)
                road_image = road_image.astype(np.float32) / 255.0
                road_image = np.reshape(road_image, (-1, 32, 32, 3))
                pred = model.predict(road_image, verbose=0)
                y_road = classes[np.argmax(pred[0])]
            except Exception:
                continue

            if use_hw and start_flag:
                if y_road == '직진':
                    motor.forward(speed=run_speed / 100)
                    if last_direction is None:
                        servo.angle = kit_middle
                    elif last_direction == 'Left':
                        servo.angle = kit_middle + adjust_angle
                    elif last_direction == 'Right':
                        servo.angle = kit_middle - adjust_angle
                elif y_road == '좌회전':
                    motor.forward(speed=run_speed / 100)
                    servo.angle = kit_middle + turn_angle
                    last_direction = 'Right'
                elif y_road == '우회전':
                    motor.forward(speed=run_speed / 100)
                    servo.angle = kit_middle - turn_angle
                    last_direction = 'Left'

            write_live_data(dist_cm=dist_cm if tof else None, pred=y_road, light=traffic_light_color)

            if cam_image is not None:
                try:
                    disp = cam_image.copy()
                    cv2.putText(disp, f'Pred: {y_road}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(disp, f'Light: {traffic_light_color or "-"}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    status = '주행중' if start_flag else ('정지 (PC에서 시작)' if params_file else '정지 (a:시작)')
                    cv2.putText(disp, f'Status: {status}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    cv2.imshow('AI Run', disp)
                except Exception:
                    pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if not params_file:
                if key == ord('a'):
                    start_flag = True
                    print('[로컬] 주행 시작')
                if key == ord('h'):
                    start_flag = False
                    if use_hw and motor is not None:
                        motor.stop()
                    print('[로컬] 주행 정지')
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if use_hw and motor is not None and servo is not None:
            motor.stop()
            servo.angle = kit_middle
        if tof is not None:
            try:
                tof.stop_ranging()
            except Exception:
                pass
        print('AI 종료')


# ==================== 메인 ====================
def main():
    parser = argparse.ArgumentParser(description='S1 도로분류')
    parser.add_argument('mode', choices=['setup', 'collect', 'process', 'train', 'run', 'full'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--camera', type=int, default=70)
    parser.add_argument('--csv', default='훈련데이터.csv')
    parser.add_argument('--block-dir', default=None)
    parser.add_argument('--speed', type=int, default=100)
    parser.add_argument('--params-file', default=None, help='원격 제어용 (PC→Pi)')
    parser.add_argument('--resume', action='store_true', help='이전 체크포인트에서 학습 재개')
    args = parser.parse_args()
    
    if args.mode == 'setup':
        setup_data_collection()
    elif args.mode == 'collect':
        setup_data_collection()
        collect_data(args.camera)
    elif args.mode == 'process':
        create_csv(args.csv)
        decalcom_augment()
        create_csv(args.csv)
        analysis_csv(args.csv)
        upsampling_csv(args.csv)
        analysis_csv(args.csv)
        print('[처리 완료]')
    elif args.mode == 'train':
        train_model(epochs=args.epochs, batch_size=args.batch, csv_name=args.csv, first_training=not args.resume)
    elif args.mode == 'run':
        # S1_main 내장 block 사용 (block 폴더가 있으면 신호등/거리센서 활성화)
        block_dir = args.block_dir or os.path.join(PROJECT_PATH, 'block')
        if not os.path.isdir(block_dir) or not os.path.isfile(os.path.join(block_dir, 'tflite_detectorlib.py')):
            block_dir = None
        run_ai(camera_id=args.camera, speed=args.speed, block_dir=block_dir, params_file=args.params_file)
    elif args.mode == 'full':
        create_csv(args.csv)
        decalcom_augment()
        create_csv(args.csv)
        analysis_csv(args.csv)
        upsampling_csv(args.csv)
        train_model(epochs=args.epochs, batch_size=args.batch, csv_name=args.csv, first_training=not args.resume)
        print('full 완료')


if __name__ == '__main__':
    main()
