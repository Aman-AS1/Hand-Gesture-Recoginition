#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import copy
import csv
import itertools
import argparse
from collections import deque, Counter

import numpy as np
import mediapipe as mp

from model import KeyPointClassifier, PointHistoryClassifier
from utils import CvFpsCalc


# ------------------------------------------------------------
# Helper Classes
# ------------------------------------------------------------

class HandDetector:
    """Handles MediaPipe hand detection + landmark extraction."""

    def __init__(self, static_image_mode, min_det_conf, min_track_conf):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

    def detect(self, image):
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        return results


class Preprocessor:
    """Processes raw landmark points into normalized classifier inputs."""

    @staticmethod
    def calc_bounding_rect(image, landmarks):
        h, w = image.shape[0], image.shape[1]
        landmark_array = np.array([(min(int(l.x * w), w - 1),
                                    min(int(l.y * h), h - 1))
                                   for l in landmarks.landmark])
        x, y, w_box, h_box = cv.boundingRect(landmark_array)
        return [x, y, x + w_box, y + h_box]

    @staticmethod
    def calc_landmark_list(image, landmarks):
        h, w = image.shape[:2]
        return [[min(int(l.x * w), w - 1), min(int(l.y * h), h - 1)]
                for l in landmarks.landmark]

    @staticmethod
    def pre_process_landmarks(landmarks):
        tmp = copy.deepcopy(landmarks)

        base_x, base_y = tmp[0]
        for p in tmp:
            p[0] -= base_x
            p[1] -= base_y

        # flatten
        tmp = list(itertools.chain.from_iterable(tmp))

        # normalize
        max_val = max(list(map(abs, tmp)))
        if max_val == 0:
            max_val = 1

        return [n / max_val for n in tmp]

    @staticmethod
    def pre_process_history(image, history):
        if len(history) == 0 or history[0] == [0, 0]:
            return []  # nothing to process yet

        h, w = image.shape[:2]
        tmp = copy.deepcopy(history)

        base_x, base_y = tmp[0]

        for p in tmp:
            p[0] = (p[0] - base_x) / w
            p[1] = (p[1] - base_y) / h

        return list(itertools.chain.from_iterable(tmp))

    # @staticmethod
    # def pre_process_history(image, history):
    #     h, w = image.shape[:2]
    #     tmp = copy.deepcopy(history)
    #
    #     base_x, base_y = tmp[0]
    #
    #     for p in tmp:
    #         p[0] = (p[0] - base_x) / w
    #         p[1] = (p[1] - base_y) / h
    #
    #     return list(itertools.chain.from_iterable(tmp))


class DataLogger:
    """Responsible for storing training data into CSV files."""

    def __init__(self):
        self.keypoint_csv = 'model/keypoint_classifier/keypoint.csv'
        self.history_csv = 'model/point_history_classifier/point_history.csv'

    def log(self, label, mode, landmarks, history):
        if mode == 1 and 0 <= label <= 9:
            with open(self.keypoint_csv, 'a', newline="") as f:
                csv.writer(f).writerow([label, *landmarks])

        elif mode == 2 and 0 <= label <= 9:
            with open(self.history_csv, 'a', newline="") as f:
                csv.writer(f).writerow([label, *history])


class Drawer:
    """Handles all drawing: bounding box, landmarks, FPS, info."""

    @staticmethod
    def draw_bounding_rect(image, rect):
        x1, y1, x2, y2 = rect
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 1)
        return image

    @staticmethod
    def draw_point_history(image, history):
        for i, p in enumerate(history):
            if p != [0, 0]:
                cv.circle(image, tuple(p), 1 + int(i / 2), (152, 251, 152), 2)
        return image

    @staticmethod
    def draw_info(image, fps, mode, label):
        cv.putText(image, f"FPS: {fps}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if mode:
            cv.putText(image, f"MODE: {mode}", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if label != -1:
                cv.putText(image, f"Label: {label}", (10, 90),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return image


# ------------------------------------------------------------
# Main Application Class
# ------------------------------------------------------------

class HandGestureApp:

    def __init__(self, args):
        self.cap = cv.VideoCapture(args.device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

        self.detector = HandDetector(
            static_image_mode=args.use_static_image_mode,
            min_det_conf=args.min_detection_confidence,
            min_track_conf=args.min_tracking_confidence
        )

        self.pre = Preprocessor()
        self.logger = DataLogger()

        self.key_classifier = KeyPointClassifier()
        self.history_classifier = PointHistoryClassifier()

        self.point_history = deque(maxlen=16)
        self.finger_history = deque(maxlen=16)

        self.fps_calc = CvFpsCalc(buffer_len=10)

        # Load gesture labels
        self.key_labels = self._load_labels(
            "model/keypoint_classifier/keypoint_classifier_label.csv")
        self.hist_labels = self._load_labels(
            "model/point_history_classifier/point_history_classifier_label.csv")

        self.mode = 0
        self.current_label = -1

    @staticmethod
    def _load_labels(path):
        with open(path, encoding="utf-8-sig") as f:
            return [row[0] for row in csv.reader(f)]

    def _process_key(self, key):
        if 48 <= key <= 57:     # '0'–'9'
            self.current_label = key - 48
        if key == ord('n'):
            self.mode = 0
        elif key == ord('k'):
            self.mode = 1
        elif key == ord('h'):
            self.mode = 2

    # ------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------
    def run(self):
        while True:
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            self._process_key(key)

            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            debug = frame.copy()

            results = self.detector.detect(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness):

                    rect = self.pre.calc_bounding_rect(debug, hand_landmarks)
                    lm_list = self.pre.calc_landmark_list(debug, hand_landmarks)

                    processed_lm = self.pre.pre_process_landmarks(lm_list)
                    processed_hist = self.pre.pre_process_history(debug, self.point_history)

                    # Logging if in logging mode
                    self.logger.log(self.current_label, self.mode, processed_lm, processed_hist)

                    sign_id = self.key_classifier(processed_lm)

                    # Track fingertip for motion gestures
                    if sign_id == 2:   # "point" gesture
                        self.point_history.append(lm_list[8])
                    else:
                        self.point_history.append([0, 0])

                    # Dynamic gesture classification
                    if len(processed_hist) == 32:
                        hist_id = self.history_classifier(processed_hist)
                        self.finger_history.append(hist_id)
                        most_common = Counter(self.finger_history).most_common(1)[0][0]
                    else:
                        most_common = 0

                    Drawer.draw_bounding_rect(debug, rect)
                    Drawer.draw_point_history(debug, self.point_history)

                    text = f"{handedness.classification[0].label}:{self.key_labels[sign_id]}"
                    cv.putText(debug, text, (rect[0], rect[1] - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv.putText(debug, f"Gesture: {self.hist_labels[most_common]}",
                               (10, 140), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                               (255, 255, 255), 2)

            # Draw general info
            fps = self.fps_calc.get()
            Drawer.draw_info(debug, fps, self.mode, self.current_label)

            cv.imshow("Hand Gesture Recognition (OOP)", debug)

        self.cap.release()
        cv.destroyAllWindows()


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    app = HandGestureApp(args)
    app.run()
