import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

# =========================
# Configuration
# =========================
VIDEO_PATH = "workshop_video.mp4"  # Change to 0 for webcam
MODEL_PATH = "yolo11s.pt"  # Use appropriate YOLOv8 model if needed

# Line coordinates (manually obtained via mouse hover)
ENTRY_LINE = ((650, 350), (950, 450))  # Red Line
EXIT_LINE = ((623, 366), (911, 465))  # Pink Line

# Threshold for detecting proximity to line
LINE_THRESH = 15

# =========================
# Helper Functions
# =========================


def get_distance_from_line(point, line_start, line_end, thresh):
    """Returns perpendicular distance from a point to a line segment."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    if x1 == x2:
        return abs(px - x1) <= thresh and min(y1, y2) <= py <= max(y1, y2)
    elif y1 == y2:
        return abs(py - y1) <= thresh and min(x1, x2) <= px <= max(x1, x2)
    else:
        num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return num / den <= thresh


def is_near_line(point, line_start, line_end, thresh=LINE_THRESH):
    return get_distance_from_line(point, line_start, line_end, thresh)


# =========================
# Entry Point
# =========================


def main():
    # Load YOLO model
    model = YOLO(MODEL_PATH)
    class_names = model.model.names

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    # Tracking state
    entry_ids = set()
    exit_ids = set()
    status_map = {}

    frame_count = 0

    # Optional: Mouse callback to get coordinates
    def show_mouse_position(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"Mouse at: ({x}, {y})")

    cv2.namedWindow("GateGuard AI")
    cv2.setMouseCallback("GateGuard AI", show_mouse_position)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))

        results = model.track(frame, persist=True, classes=0)  # Only person class (0)

        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                label = class_names[class_id]
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                center = (cx, cy)

                near_entry = is_near_line(center, *ENTRY_LINE)
                near_exit = is_near_line(center, *EXIT_LINE)

                if track_id not in status_map:
                    if near_entry:
                        status_map[track_id] = "entered"
                        entry_ids.add(track_id)
                    elif near_exit:
                        status_map[track_id] = "exited"
                        exit_ids.add(track_id)

                # Draw bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (255, 255, 255), -1)
                cvzone.putTextRect(
                    frame, f"ID: {track_id}", (x1, y2), scale=1, thickness=2
                )
                cvzone.putTextRect(
                    frame, f"{label}", (x1, y1 - 20), scale=1, thickness=2
                )

        # Draw entry/exit lines
        cv2.line(frame, ENTRY_LINE[0], ENTRY_LINE[1], (0, 0, 255), 3)
        cv2.line(frame, EXIT_LINE[0], EXIT_LINE[1], (255, 0, 255), 3)

        # Show stats
        cvzone.putTextRect(
            frame, f"Entered: {len(entry_ids)}", (50, 60), scale=2, thickness=2
        )
        cvzone.putTextRect(
            frame, f"Exited: {len(exit_ids)}", (50, 120), scale=2, thickness=2
        )
        cvzone.putTextRect(
            frame,
            f"Inside: {len(entry_ids) - len(exit_ids)}",
            (50, 180),
            scale=2,
            thickness=2,
        )

        cv2.imshow("GateGuard AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    main()
