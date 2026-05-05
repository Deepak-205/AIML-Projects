import cv2
import time

print("Starting camera diagnostic...")

# Try multiple backends
backends = [
    cv2.CAP_DSHOW,
    cv2.CAP_MSMF,
    cv2.CAP_ANY
]

for backend in backends:
    print(f"\nTrying backend: {backend}")
    cap = cv2.VideoCapture(0, backend)
    time.sleep(1)

    if cap.isOpened():
        print("Camera opened successfully!")
        ret, frame = cap.read()
        print("Frame read:", ret)

        if ret:
            cv2.imshow("Camera Test", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            cap.release()
            print("Backend works:", backend)
            break
        else:
            print("Opened but no frame.")
    else:
        print("Failed to open camera.")

    cap.release()

print("Diagnostic finished.")