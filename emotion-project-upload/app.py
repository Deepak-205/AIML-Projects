from flask import Flask, jsonify, send_from_directory, Response, request
import threading
import os
import time
import cv2
import emotion_engine as cam

app = Flask(__name__)

# =========================
# 🔥 PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSIC_FOLDER = os.path.join(BASE_DIR, "music")

print("BASE DIR:", BASE_DIR)
print("MUSIC DIR:", MUSIC_FOLDER)

# =========================
# 🎛️ MODE CONTROL
# =========================
MODE = "FAST"  # default

@app.route("/set_mode", methods=["POST"])
def set_mode():
    global MODE
    data = request.get_json()
    MODE = data.get("mode", "FAST")
    print(f"[MODE] Switched to: {MODE}")
    return {"status": "ok", "mode": MODE}

@app.route("/get_mode")
def get_mode():
    return {"mode": MODE}

# =========================
# 🎵 Serve music files
# =========================
@app.route("/music/<path:filename>")
def serve_music(filename):
    return send_from_directory(MUSIC_FOLDER, filename)

# =========================
# 📂 PLAYLIST API
# =========================
@app.route("/playlist/<emotion>")
def get_playlist(emotion):
    folder_path = os.path.join(MUSIC_FOLDER, emotion)

    if not os.path.exists(folder_path):
        return jsonify([])

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".mp3")
    ]

    return jsonify(files)

# =========================
# 🌐 HOME
# =========================
from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")

# =========================
# 📊 Emotion API
# =========================
@app.route("/emotion")
def emotion():
    """
    Expected format from cam.latest_emotion:
    {
        "emotion": "happy",
        "live": "happy"
    }
    """
    return jsonify(cam.latest_emotion)

# =========================
# 🌙 NIGHT VISION
# =========================
@app.route("/toggle_night")
def toggle_night():
    cam.NIGHT_VISION = not cam.NIGHT_VISION
    return "OK"

# =========================
# 🎥 VIDEO STREAM (OPTIMIZED)
# =========================
def generate_frames():
    while True:
        frame = cam.get_frame()

        if frame is None:
            time.sleep(0.05)
            continue

        # Resize for performance
        frame = cv2.resize(frame, (480, 360))

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # ~25 FPS
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =========================
# 🎥 Camera thread
# =========================
def run_camera():
    cam.main()

# =========================
# 🚀 MAIN
# =========================
if __name__ == "__main__":
    print("🚀 Starting Emotion Music System...")

    # Run camera in background
    threading.Thread(target=run_camera, daemon=True).start()

    # Start Flask
    app.run(debug=False, use_reloader=False)