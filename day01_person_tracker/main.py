import cv2
from ultralytics import YOLO


# -----------------------------
# 설정
# -----------------------------
MODEL_NAME = "yolo26n.pt"
CAMERA_ID = 0

CONFIDENCE = 0.5
DEAD_ZONE = 0.15

SERVO_CENTER = 90
SERVO_MIN = 30
SERVO_MAX = 150

MAX_SERVO_OFFSET = 60

# 0 ~ 1
# 클수록 새 값을 빠르게 반영
SMOOTHING_ALPHA = 0.9


# -----------------------------
# 서보 컨트롤러
# -----------------------------
def calculate_servo_angle(error_x):

    if abs(error_x) < DEAD_ZONE:
        error_x = 0

    servo_angle = SERVO_CENTER + error_x * MAX_SERVO_OFFSET

    servo_angle = max(
        SERVO_MIN,
        min(SERVO_MAX, servo_angle)
    )

    return int(servo_angle)


# -----------------------------
# AI 모델
# -----------------------------
model = YOLO(MODEL_NAME)


# -----------------------------
# 카메라
# -----------------------------
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")


# 이전 프레임에서 사용한 값
smoothed_error = 0.0


while True:

    success, frame = cap.read()

    if not success:
        print("카메라 프레임을 읽지 못했습니다.")
        break

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    frame_center_x = width // 2
    frame_center_y = height // 2

    servo_angle = SERVO_CENTER

    # -----------------------------
    # AI inference
    # -----------------------------
    result = model(
        frame,
        verbose=False,
        conf=CONFIDENCE,
        imgsz=480
    )[0]

    people = []

    # -----------------------------
    # 사람 탐지
    # -----------------------------
    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name != "person":
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].cpu().tolist()
        )

        area = (x2 - x1) * (y2 - y1)

        people.append(
            (area, x1, y1, x2, y2)
        )

    # -----------------------------
    # 가장 큰 사람 선택
    # -----------------------------
    if people:

        target = max(
            people,
            key=lambda person: person[0]
        )

        _, x1, y1, x2, y2 = target

        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2

        # 원본 error
        raw_error = (
            person_center_x - frame_center_x
        ) / (width / 2)

        # -----------------------------
        # Smoothing
        # -----------------------------
        smoothed_error = (
            SMOOTHING_ALPHA * raw_error
            + (1 - SMOOTHING_ALPHA) * smoothed_error
        )

        # -----------------------------
        # Controller
        # -----------------------------
        servo_angle = calculate_servo_angle(smoothed_error)

        # 방향
        if smoothed_error < -DEAD_ZONE:
            direction = "LEFT"

        elif smoothed_error > DEAD_ZONE:
            direction = "RIGHT"

        else:
            direction = "CENTER"

        # 사람 박스
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # 사람 중심
        cv2.circle(
            frame,
            (person_center_x, person_center_y),
            7,
            (0, 0, 255),
            -1
        )

        # 화면 중심 → 사람 중심
        cv2.arrowedLine(
            frame,
            (frame_center_x, frame_center_y),
            (person_center_x, person_center_y),
            (255, 0, 0),
            3
        )

        # 원본 error
        cv2.putText(
            frame,
            f"RAW: {raw_error:.2f}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # smoothing 된 error
        cv2.putText(
            frame,
            f"SMOOTH: {smoothed_error:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{direction}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    # -----------------------------
    # 가상 서보 표시
    # -----------------------------
    cv2.putText(
        frame,
        f"SERVO: {servo_angle} deg",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.line(
        frame,
        (frame_center_x, 0),
        (frame_center_x, height),
        (255, 255, 0),
        2
    )

    cv2.imshow(
        "Physical AI - Smooth Servo Tracker",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()