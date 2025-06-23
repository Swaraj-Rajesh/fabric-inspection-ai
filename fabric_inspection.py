import cv2
import queue
import threading
import signal
import time
import numpy as np
import lgpio
from tensorflow.keras.models import load_model
from math import hypot

try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
    print("Using Picamera2")
except ImportError:
    USE_PICAMERA = False
    print("Using OpenCV")

frame_queue = queue.Queue(maxsize=3)
stop_event = threading.Event()

# Load your trained model
model = load_model("../models/ai.h5")  # Adjust path if needed

# Stepper motor configuration
PUL_PIN = 23
DIR_PIN = 24
ENA_PIN = 25

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, PUL_PIN)
lgpio.gpio_claim_output(h, DIR_PIN)
lgpio.gpio_claim_output(h, ENA_PIN)
lgpio.gpio_write(h, ENA_PIN, 0)
lgpio.gpio_write(h, DIR_PIN, 1)

motor_running = True
pulse_duration = 0.0009

detected_defects = []

def is_new_defect(center, threshold=80):
    for old_center in detected_defects:
        if hypot(center[0] - old_center[0], center[1] - old_center[1]) < threshold:
            return False
    return True

def step_motor():
    lgpio.gpio_write(h, PUL_PIN, 1)
    time.sleep(pulse_duration)
    lgpio.gpio_write(h, PUL_PIN, 0)
    time.sleep(pulse_duration)

def motor_control():
    while not stop_event.is_set():
        if motor_running:
            step_motor()
        else:
            time.sleep(0.1)

motor_thread = threading.Thread(target=motor_control, daemon=True)
motor_thread.start()

def capture_frames():
    print("Starting capture_frames()")
    if USE_PICAMERA:
        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (640, 480)
        picam2.preview_configuration.main.format = "RGB888"
        picam2.configure("preview")
        picam2.start()
        time.sleep(1)
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while not stop_event.is_set():
        if USE_PICAMERA:
            frame = picam2.capture_array()
        else:
            ret, frame = cap.read()
            if not ret:
                continue

        if not frame_queue.full():
            frame_queue.put(frame)

    if not USE_PICAMERA:
        cap.release()

def process_frames():
    global motor_running

    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        edges = cv2.Canny(frame, 100, 200)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        defect_result, major_defect_found = ai_inference(frame)

        motor_anim = generate_motor_animation(major_defect_found, frame.shape[1])
        fabric_status = generate_fabric_status(major_defect_found, frame.shape[1])
        combined_top = np.hstack([edges_colored, defect_result])
        combined_bottom = np.hstack([motor_anim, fabric_status])
        final_display = np.vstack([combined_top, combined_bottom])

        cv2.imshow("Fabric Inspection System", final_display)

        if major_defect_found and motor_running:
            threading.Thread(target=stop_motor_for_defect, daemon=True).start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()

    cv2.destroyAllWindows()

def ai_inference(frame):
    global detected_defects
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame.copy()
    major_defect_found = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 4:
                center = (int(x), int(y))
                if is_new_defect(center):
                    print(f"🚨 Major defect at {center}, stopping motor.")
                    detected_defects.append(center)
                    major_defect_found = True
                    cv2.circle(output, center, int(radius)+15, (0, 0, 255), 6)

    return output, major_defect_found

def stop_motor_for_defect():
    global motor_running
    print(" Major defect detected. Pausing motor for 3 seconds...")
    motor_running = False

    PUMP_IN1 = 14
    PUMP_IN2 = 15
    PUMP_ENA = 18

    lgpio.gpio_claim_output(h, PUMP_IN1, 0)
    lgpio.gpio_claim_output(h, PUMP_IN2, 0)
    lgpio.gpio_claim_output(h, PUMP_ENA, 0)

    def start_pump(duration=1):
        print("Pump ON")
        lgpio.gpio_write(h, PUMP_IN1, 1)
        lgpio.gpio_write(h, PUMP_IN2, 0)
        lgpio.tx_pwm(h, PUMP_ENA, 1000, 100)
        time.sleep(duration)
        stop_pump()

    def stop_pump():
        print("Pump OFF")
        lgpio.gpio_write(h, PUMP_IN1, 0)
        lgpio.gpio_write(h, PUMP_IN2, 0)
        lgpio.tx_pwm(h, PUMP_ENA, 0, 0)

    start_pump(1)

    lgpio.gpio_write(h, ENA_PIN, 1)
    time.sleep(3)
    print("▶ Motor resuming...")
    lgpio.gpio_write(h, ENA_PIN, 0)
    motor_running = True

def generate_motor_animation(defect, width):
    anim = np.zeros((200, width, 3), dtype=np.uint8)
    color = (0, 255, 0) if not defect else (0, 0, 255)
    cv2.circle(anim, (width // 2, 100), 50, color, -1)
    return anim

def generate_fabric_status(defect, width):
    status_img = np.zeros((200, width, 3), dtype=np.uint8)
    text = "FABRIC OK" if not defect else "MAJOR DEFECT!"
    color = (0, 255, 0) if not defect else (0, 0, 255)
    cv2.putText(status_img, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    return status_img

def signal_handler(sig, frame):
    print("CTRL+C pressed. Exiting...")
    stop_event.set()

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("Launching Fabric Inspection System...")
    threading.Thread(target=capture_frames, daemon=True).start()
    process_frames()
