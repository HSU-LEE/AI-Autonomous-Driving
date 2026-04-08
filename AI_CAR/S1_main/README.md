# S1 도로분류 AI - 라즈베리 파이 단독 배치

**이 폴더만** 라즈베리 파이로 복사하면 됩니다. `/home/pi/S1_main`에 두세요.

## 빠른 시작

```bash
cd /home/pi/S1_main
pip install -r requirements.txt
python S1_main.py setup
python S1_main.py collect --camera 70
python S1_main.py process
python S1_main.py train --epochs 10
python S1_main.py run --camera 70
```

## 폴더 구조 (S1_main만 복사)

```
S1_main/
├── S1_main.py
├── names.json
├── kitinfo.json        # cp kitinfo.json.example kitinfo.json
├── requirements.txt
├── block/              # 내장 (신호등/거리센서)
│   ├── tflite_detectorlib.py
│   ├── traffic_color_check.py
│   ├── cameralib.py
│   ├── VL53L0X.py
│   ├── traffic_sign_20epochs_person2.tflite  # 별도 추가
│   └── vl53l0x_python.so                     # Pi 거리센서 시 별도 추가
├── remote/             # PC→Pi 원격 제어
│   ├── S1_remote_server.py   # Pi에서 실행
│   └── S1_remote_client.py   # PC에서 실행
├── data/
└── model/
```

## PC 원격 제어 (자율주행차 완벽 제어)

**라즈베리 파이에서:**
```bash
pip install flask flask-cors
python remote/S1_remote_server.py
# 화면에 표시된 http://192.168.x.x:5001 주소 확인
```

**PC에서:**
```bash
pip install Pillow
python remote/S1_remote_client.py
# Pi IP 입력 후 [연결] → Setup, 수집, 처리, 학습, AI실행 모두 PC에서 제어
```

## 선택 파일 (block 폴더)

- **traffic_sign_20epochs_person2.tflite**: 신호등 탐지 모델 (없으면 기본 주행만)
- **vl53l0x_python.so**: 거리센서 사용 시 (없으면 거리센서 미사용)
