# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# S1_main 내장용 - block 폴더 내에서 동작
import os
import cv2
from tflite_support.task import core, processor, vision
import numpy as np

_BLOCK_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TFLITE_MODEL = os.path.join(_BLOCK_DIR, 'traffic_sign_20epochs_person2.tflite')
_TFLITE_MODEL_PATH = os.environ.get('TFLITE_SIGN_MODEL', _DEFAULT_TFLITE_MODEL)

_MARGIN = 10
_ROW_SIZE = 10
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)

detector = None
if os.path.isfile(_TFLITE_MODEL_PATH):
    try:
        base_options = core.BaseOptions(file_name=_TFLITE_MODEL_PATH, use_coral=False, num_threads=4)
        detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.6)
        options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
        detector = vision.ObjectDetector.create_from_options(options)
    except Exception as e:
        print(f'[tflite_detectorlib] 모델 로드 실패: {e}')
else:
    print(f'[tflite_detectorlib] 모델 파일 없음: {_TFLITE_MODEL_PATH}')

def visualize(image, detection_result):
    if len(detection_result.detections) < 1:
        return None, (None, None, None, None)
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        category = detection.categories[0]
        category_name = category.category_name
        return category_name, (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
    return None, (None, None, None, None)

def detect(image):
    if detector is None:
        return None, (None, None, None, None)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    detection_result = detector.detect(input_tensor)
    color, (x, y, width, height) = visualize(image, detection_result)
    if color == "Red":
        color = 'red'
    if color == 'Green':
        color = 'green'
    return color, (x, y, width, height)

def detect1(image):
    rv1 = (None, None, None, None, None)
    rv2 = (None, None, None, None, None)
    rv3 = (None, None, None, None, None)
    rv4 = (None, None, None, 0, 0)
    if detector is None:
        return [image.copy(), rv1, rv2, rv3, rv4]
    origin = image.copy()
    rgb_image = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    detection_result = detector.detect(input_tensor)
    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        bbox = detection.bounding_box
        if category_name == '70':
            rv1 = (70, bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
        if category_name == '30':
            rv1 = (30, bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
        if category_name == 'Green':
            rv2 = ('green', bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
        if category_name == 'Red':
            rv2 = ('red', bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
        if category_name == 'Off':
            rv2 = ('off', bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
        if category_name == 'Stop':
            rv3 = ('stop', bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
        if category_name == 'person':
            rv4 = ('person', bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (bbox.origin_x - 5, bbox.origin_y - 5)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    return [image, rv1, rv2, rv3, rv4]
