import pdb
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
# module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" #frcnn
detector = hub.load(module_handle).signatures['default']
print("Mobilenet model loaded")


def reshape_input(img):
    return tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

def detect_mobilenet(img,detector=detector):
    img = reshape_input(img)
    result = detector(img)
    # result = {key: value.numpy() for key, value in result.items()}
    return result


def draw(detections,img, im_width, im_height,):
    detections_as_xyxy = detections['detection_boxes']
    detections_as_conf = detections['detection_scores']
    detections_as_cls = detections['detection_class_entities']
    # pdb.set_trace()

    for box, conf, cls in zip(detections_as_xyxy, detections_as_conf, detections_as_cls):
        if conf < 0.2:
            continue
        # pdb.set_trace()
        ymin, xmin, ymax, xmax = tuple(box)
        (left, right, top, bottom) = (int(xmin * im_height), int(xmax * im_height),
                                      int(ymin * im_width), int(ymax * im_width))


        cv2.rectangle(img, (left, top), (right,bottom), (0, 0, 255), 3)
        cls_name = cls.numpy().decode('utf-8')
        print(cls_name)

    return img


def mobilenet_out_detections(
    detections, im_width, im_height, track_points: str = "bbox",  # bbox or centroid
         ):
    """convert detections_as_xywh to norfair detections"""
    norfair_detections = []

    if track_points == "bbox":
        detections_class_ids = detections['detection_class_labels']
        detections_as_xyxy = detections['detection_boxes']
        detections_as_conf = detections['detection_scores']
        detections_as_cls = detections['detection_class_entities']
        # pdb.set_trace()

        for box, conf, cls, cls_id in zip(detections_as_xyxy, detections_as_conf, detections_as_cls, detections_class_ids):
            if conf < 0.2:
                continue
            # pdb.set_trace()
            ymin, xmin, ymax, xmax = tuple(box)
            (left, right, top, bottom) = (int(xmin * im_height), int(xmax * im_height),
                                          int(ymin * im_width), int(ymax * im_width))
            bbox = np.array(
                [
                    left, top,
                    right, bottom

                ]
            )
            conf = conf.numpy()
            # scores = np.array(conf)
            cls_name = cls.numpy().decode('utf-8')
            label = cls_name
            norfair_detections.append([bbox, conf, label])

    return norfair_detections


def draw_boxes(frame, detections, idx):
    # pdb.set_trace()
    print()
    for id in idx:
        item = detections[id]
        x,y,x2,y2 = item[0]
        cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(frame, f"{item[2]}:{int(item[1] * 100)}%", (int(x) + 10, int(y)), 1, 2, (0, 255, 0))
    return frame