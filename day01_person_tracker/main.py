import cv2
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "yolo26n.pt"
CAMERA_ID = 0

CONFIDENCE = 0.5

DEAD_ZONE = 0.15
SMOOTHING_ALPHA = 0.2

SERVO_CENTER = 90
SERVO_MIN = 30
SERVO_MAX = 150

MAX_SERVO_OFFSET = 60
MAX_SERVO_STEP = 3


# ============================================================
# CONTROL FUNCTIONS
# ============================================================

def calculate_target_angle(error_x):

    # 중앙 근처라면 움직이지 않는다.
    if abs(error_x) < DEAD_ZONE:
        error_x = 0

    target_angle = (
        SERVO_CENTER
        + error_x * MAX_SERVO_OFFSET
    )

    target_angle = max(
        SERVO_MIN,
        min(SERVO_MAX, target_angle)
    )

    return int(target_angle)


def move_servo_toward(current_angle, target_angle):

    difference = target_angle - current_angle

    # 목표가 아주 가까우면 바로 도착
    if abs(difference) <= MAX_SERVO_STEP:
        return target_angle

    # 오른쪽 방향으로 이동
    if difference > 0:
        return current_angle + MAX_SERVO_STEP

    # 왼쪽 방향으로 이동
    return current_angle - MAX_SERVO_STEP


# ============================================================
# AI MODEL
# ============================================================

model = YOLO(MODEL_NAME)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")


# ============================================================
# ROBOT STATE
# ============================================================

smoothed_error = 0.0
current_servo_angle = SERVO_CENTER


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("카메라 프레임을 읽지 못했습니다.")
        break

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    frame_center_x = width // 2
    frame_center_y = height // 2

    # 기본 상태
    tracking = False

    raw_error = 0.0
    target_servo_angle = SERVO_CENTER

    # --------------------------------------------------------
    # PERCEPTION
    # --------------------------------------------------------

    result = model(
        frame,
        verbose=False,
        conf=CONFIDENCE,
        imgsz=480
    )[0]

    people = []

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

    # --------------------------------------------------------
    # TARGET SELECTION
    # --------------------------------------------------------

    if people:

        tracking = True

        target = max(
            people,
            key=lambda person: person[0]
        )

        _, x1, y1, x2, y2 = target

        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2

        # ----------------------------------------------------
        # POSITION ERROR
        # ----------------------------------------------------

        raw_error = (
            person_center_x - frame_center_x
        ) / (width / 2)

        # ----------------------------------------------------
        # SMOOTHING
        # ----------------------------------------------------

        smoothed_error = (
            SMOOTHING_ALPHA * raw_error
            + (1 - SMOOTHING_ALPHA)
            * smoothed_error
        )

        # ----------------------------------------------------
        # CONTROLLER
        # ----------------------------------------------------

        target_servo_angle = calculate_target_angle(
            smoothed_error
        )

        current_servo_angle = move_servo_toward(
            current_servo_angle,
            target_servo_angle
        )

        # ----------------------------------------------------
        # DRAW PERSON
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (person_center_x, person_center_y),
            7,
            (0, 0, 255),
            -1
        )

        cv2.arrowedLine(
            frame,
            (frame_center_x, frame_center_y),
            (person_center_x, person_center_y),
            (255, 0, 0),
            3
        )

    # --------------------------------------------------------
    # NO TARGET
    # --------------------------------------------------------

    else:

        # 사람을 잃었을 때 error를 천천히 0으로 복귀
        smoothed_error = (
            (1 - SMOOTHING_ALPHA)
            * smoothed_error
        )
        
        current_servo_angle = move_servo_toward(
        current_servo_angle,
        SERVO_CENTER
    )
    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    cv2.line(
        frame,
        (frame_center_x, 0),
        (frame_center_x, height),
        (255, 255, 0),
        2
    )

    if tracking:
        status = "TRACKING"
    else:
        status = "NO PERSON"

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0) if tracking else (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"RAW ERROR: {raw_error:.2f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"SMOOTH ERROR: {smoothed_error:.2f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"TARGET ANGLE: {target_servo_angle}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"CURRENT ANGLE: {current_servo_angle}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Physical AI - Person Tracking System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()