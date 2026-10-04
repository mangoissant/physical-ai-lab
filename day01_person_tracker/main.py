import cv2
from ultralytics import YOLO


# -----------------------------
# 설정
# -----------------------------
MODEL_NAME = "yolo26n.pt"
CAMERA_ID = 0
CONFIDENCE = 0.5
DEAD_ZONE = 0.15


# -----------------------------
# AI 모델 로드
# -----------------------------
model = YOLO(MODEL_NAME)


# -----------------------------
# 카메라 열기
# -----------------------------
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")


while True:
    success, frame = cap.read()

    if not success:
        print("카메라 프레임을 읽지 못했습니다.")
        break

    # 거울처럼 보이도록 좌우 반전
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    frame_center_x = width // 2
    frame_center_y = height // 2

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
    # 탐지 결과 중 사람만 찾기
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
    # 가장 크게 보이는 사람 추적
    # -----------------------------
    if people:

        target = max(people, key=lambda person: person[0])

        _, x1, y1, x2, y2 = target

        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2

        # 화면 중심 기준 오차
        error_x = (
            person_center_x - frame_center_x
        ) / (width / 2)

        # -----------------------------
        # 방향 결정
        # -----------------------------
        if error_x < -DEAD_ZONE:
            direction = "LEFT"

        elif error_x > DEAD_ZONE:
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

        # 결과 표시
        cv2.putText(
            frame,
            f"{direction}  error={error_x:.2f}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # 화면 중심점
    cv2.circle(
        frame,
        (frame_center_x, frame_center_y),
        7,
        (255, 255, 0),
        -1
    )

    cv2.imshow("Physical AI - Person Tracker", frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()