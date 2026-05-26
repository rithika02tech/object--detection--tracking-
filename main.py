import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8s.pt")

tracker = DeepSort(max_age=20)

mode = "webcam"
video_path = r"C:\Users\acer\Desktop\object detction\video.mp4"

if mode == "webcam":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Source not found")
    exit()

frame_skip = 2
frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (640, 480))

    results = model(frame, conf=0.5, verbose=False)

    detections = []

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            if confidence > 0.5:

                detections.append((
                    [x1, y1, x2 - x1, y2 - y1],
                    confidence,
                    class_name
                ))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        x1, y1, x2, y2 = map(int, track.to_ltrb())

        class_name = track.get_det_class()

        label = f"{class_name} ID:{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

    cv2.imshow("Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()