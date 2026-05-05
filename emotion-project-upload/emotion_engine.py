import cv2
import numpy as np
import time
import threading
from collections import deque

print("Starting Emotion Recognition Webcam...")

# =========================
# 🔥 SHARED STATE
# =========================
latest_emotion = {
    "emotion": "analyzing",
    "live": "analyzing",
    "confidence": 0.0
}

latest_frame = None

# =========================
# 🧠 LOAD MODEL
# TFLite first (fast), falls back to .h5 if not found
# =========================
USE_TFLITE = False
interpreter = None
model = None

import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
tflite_path = os.path.join(BASE_DIR, "models", "emotion_model.tflite")
h5_path     = os.path.join(BASE_DIR, "models", "emotion_model_esl.h5")

try:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    USE_TFLITE = True
    print("[Model] TFLite loaded ✓ (fast mode)")

except Exception as e:
    print("TFLite failed:", e)

    model = tf.keras.models.load_model(h5_path)
    USE_TFLITE = False
    print("[Model] Keras .h5 loaded ✓ (fallback)")

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# =========================
# 🧠 INFERENCE FUNCTION
# =========================
def run_inference(face_array):
    if USE_TFLITE:
        interpreter.set_tensor(input_details[0]['index'], face_array)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0]
    else:
        return model.predict(face_array, verbose=0)[0]

# =========================
# 👁️ FACE DETECTOR
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

proto_path = os.path.join(BASE_DIR, "models", "deploy.prototxt")
model_path = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")

net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# =========================
# 🧠 ENGINE CONFIG
# =========================
emotion_history = deque(maxlen=20)

last_stable_emotion = "neutral"
last_switch_time = time.time()
last_sample_time = 0

DOMINANCE_THRESHOLD = 0.6
MIN_SWITCH_DELAY = 2.0

# =========================
# 🆕 INFERENCE CONTROL
# =========================
TARGET_FPS = 10
last_inference_time = 0

# 🔥 REAL PREDICTION STORAGE
last_raw_emotion = "neutral"
last_confidence = 0.0

# =========================
# 🟡 ANALYZING
# =========================
start_time = time.time()
ANALYZE_DURATION = 3
is_analyzing = True

# =========================
# 🌙 NIGHT VISION
# =========================
NIGHT_VISION = False

# =========================
# 🔥 FIX 2: INFERENCE THREAD
# model.predict() runs in background
# camera loop never waits for it
# =========================
_infer_lock   = threading.Lock()
_infer_input  = None
_infer_result = None
_infer_ready  = False

def _inference_worker():
    global _infer_input, _infer_result, _infer_ready
    while True:
        face_arr = None
        with _infer_lock:
            if _infer_input is not None:
                face_arr = _infer_input
                _infer_input = None
        if face_arr is not None:
            preds = run_inference(face_arr)
            idx   = int(np.argmax(preds))
            with _infer_lock:
                _infer_result = (emotion_labels[idx], float(preds[idx]))
                _infer_ready  = True
        else:
            time.sleep(0.01)

threading.Thread(target=_inference_worker, daemon=True).start()
print("[Inference] Background thread started ✓")


def enhance_for_model(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

def night_vision_display(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    nv[:, :, 1] = np.clip(nv[:, :, 1] * 1.6, 0, 255)
    return nv


def main():
    global latest_emotion, latest_frame
    global last_stable_emotion, last_switch_time, last_sample_time
    global is_analyzing, NIGHT_VISION
    global last_inference_time
    global last_raw_emotion, last_confidence
    global _infer_input, _infer_result, _infer_ready

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  #always read latest frame

    time.sleep(1)

    if not cap.isOpened():
        print("Camera not accessible")
        return

    print("Camera running...")

    last_box = None
    lost_frames = 0
    MAX_LOST = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # =========================
        # 🟡 ANALYZING
        # =========================
        if is_analyzing:
            if current_time - start_time < ANALYZE_DURATION:
                latest_emotion = {
                    "emotion": "analyzing",
                    "live": "analyzing",
                    "confidence": 0.0
                }
                latest_frame = frame.copy()
                continue
            else:
                is_analyzing = False
                emotion_history.clear()
                last_switch_time = current_time

        # =========================
        # 🔆 LIGHT BOOST
        # =========================
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

        (h, w) = frame.shape[:2]

        # =========================
        # 👁️ FACE DETECTION
        # =========================
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        best_box = None
        max_area = 0

        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]

            if conf > 0.4:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                area = (x2 - x1) * (y2 - y1)

                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)

        if best_box is not None:
            last_box = best_box
            lost_frames = 0
        else:
            lost_frames += 1

        # =========================
        # 🎯 EMOTION
        # =========================
        if last_box is not None and lost_frames < MAX_LOST:

            (x1, y1, x2, y2) = last_box

            pad = int(0.25 * (y2 - y1))

            y1 = max(0, y1 - pad)
            y2 = min(frame.shape[0], y2 + pad)
            x1 = max(0, x1 - pad)
            x2 = min(frame.shape[1], x2 + pad)

            face = frame[y1:y2, x1:x2]

            if face.size != 0:

                gray = enhance_for_model(face)

                gray = cv2.resize(gray, (48, 48))
                gray = gray.astype("float32") / 255.0
                gray = np.reshape(gray, (1, 48, 48, 1))

                # =========================
                # 🔥 FIX 1: NON-BLOCKING INFERENCE
                # hand off to thread, don't wait
                # =========================
                if current_time - last_inference_time >= 1 / TARGET_FPS:
                    with _infer_lock:
                        _infer_input = gray          # thread picks this up
                    last_inference_time = current_time

                # collect result if thread finished
                with _infer_lock:
                    if _infer_ready:
                        last_raw_emotion, last_confidence = _infer_result
                        _infer_ready = False

                raw_emotion = last_raw_emotion
                confidence = last_confidence
                live_emotion = raw_emotion

                # =========================
                # 🔁 SMART BUFFER
                # =========================
                if current_time - last_sample_time >= 0.2:

                    if last_confidence > 0.6:
                        emotion_history.extend([last_raw_emotion, last_raw_emotion])

                    elif last_confidence > 0.4:
                        emotion_history.append(last_raw_emotion)

                    else:
                        emotion_history.append(last_raw_emotion)

                    last_sample_time = current_time

                # =========================
                # 🧠 STABLE
                # =========================
                if len(emotion_history) >= 10:

                    counts = {}
                    for emo in emotion_history:
                        counts[emo] = counts.get(emo, 0) + 1

                    dominant = max(counts, key=counts.get)
                    ratio = counts[dominant] / sum(counts.values())

                    if ratio >= DOMINANCE_THRESHOLD and \
                       current_time - last_switch_time > MIN_SWITCH_DELAY:

                        if dominant != last_stable_emotion:
                            last_stable_emotion = dominant
                            last_switch_time = current_time

                final_emotion = last_stable_emotion

                latest_emotion = {
                    "emotion": final_emotion,
                    "live": live_emotion,
                    "confidence": confidence
                }

                if NIGHT_VISION:
                    frame = night_vision_display(frame)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                cv2.putText(frame, f"LIVE: {live_emotion}", (x1, y1 - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                cv2.putText(frame, f"STABLE: {final_emotion}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        else:
            latest_emotion = {
                "emotion": "no_face",
                "live": "no_face",
                "confidence": 0.0
            }

        latest_frame = frame.copy()

    cap.release()


def get_frame():
    global latest_frame
    return latest_frame