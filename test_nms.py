import cv2
from ultralytics import YOLO
from mobilenet import detect_mobilenet, mobilenet_out_detections,draw_boxes
from yolo import yolo_out_detections
from mns import non_max_suppression


model = YOLO('yolov8l-world.pt')


frame = cv2.imread('bottle_frame1.jpg')
im_width,im_height = frame.shape[:2]

yolo_out = model.predict(frame)
mobilenet_out = detect_mobilenet(frame)

yolo_detections = yolo_out_detections(yolo_out)
mobilenet_detections = mobilenet_out_detections(mobilenet_out,im_width,im_height)

all_detections = yolo_detections + mobilenet_detections

all_boxes = non_max_suppression(all_detections)

out_frame = draw_boxes(frame, all_detections,all_boxes)
cv2.imwrite("out_frame.jpg", out_frame)
print("done")




