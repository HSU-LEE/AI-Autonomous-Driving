# S1_main 내장용 - 신호등 빨간불 판정
from collections import deque
import json
import os
import cv2
import tflite_detectorlib as tf_detector

que = deque([None, None, None, None, None], maxlen=5)
que1 = deque([0] * 15, maxlen=15)

def color_append(color):
    que.append(color)

def color_check():
    """빨간불 비율 60% 이상이면 True (정지 신호)"""
    if len(que) == 0:
        return False
    red_count = que.count('red')
    if red_count == 0:
        return False
    per = red_count / len(que) * 100
    return per >= 60

def red_append(size):
    que1.append(size)

def mean_size():
    return int(sum(que1) / len(que1)) if que1 else 0

def max_size():
    return int(max(que1)) if que1 else 0
