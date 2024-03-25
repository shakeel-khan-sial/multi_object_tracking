import pdb

import numpy as np
from ultralytics import YOLO
import cv2
from norfair import Detection, Tracker, Video, draw_tracked_objects
# from yolo import YOLO, yolo_detections_to_norfair_detections


tracker = Tracker(distance_function="euclidean", distance_threshold=100)

# # read image
# img = cv2.imread('traffic_city.jpg')
# # Initialize a YOLO-World model
model = YOLO('yolov8l-world.pt')  # or choose yolov8m/l-world.pt


def detect(img, model=model):
    print(img.shape)
    results = model.predict(img)
    return results[0]
    # print(dir(results[0]))
    # # pdb.set_trace()
    # boxes = results[0].boxes
    # return boxes

def draw(results):
    all_cls = results.names
    boxes = results.boxes
    img = results.orig_img
    norfair_detections = []
    for box in boxes:
        cls = all_cls[int(box.cls)]
        conf = box.conf
        xyxy = box.xyxy[0]
        x1,y1,x2,y2 = map(int,xyxy[0:4])
        image = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        image = cv2.putText(image, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        detect = Detection(np.array([x1,y1,x2,y2]))
        norfair_detections.append(detect)
        #
    tracked_objects = tracker.update(detections=norfair_detections)
    print(tracked_objects)
    draw_tracked_objects(image, tracked_objects)
        # # video.write(frame)

    return image


def detect_and_draw(img):

    result = detect(img)
    img_out = draw(result)
    # cv2.imwrite("out.jpg", img_out)
    return img_out


def my_detector(frame, model=model):
    results = model(frame)
    # pdb.set_trace()
    return results
    # pdb.set_trace()