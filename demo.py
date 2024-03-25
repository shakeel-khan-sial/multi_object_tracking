import argparse
import pdb
import cv2
from typing import List
import time
import numpy as np
from mns import non_max_suppression
from mobilenet import detect_mobilenet,mobilenet_out_detections
from draw import center, draw
from yolo import yolo_ultralytics_detections_to_norfair_detections,mobilenet_to_norfair_detections,yolo_out_detections,norfair_input
from ultralytics import YOLO

from norfair import AbsolutePaths, Paths, Tracker, Video
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator
from norfair.distances import create_normalized_mean_euclidean_distance
from main import my_detector
DISTANCE_THRESHOLD_CENTROID: float = 0.08


def inference(
    input_video: str, model: str, track_points: str, model_threshold: str, classes: List
):
    st = time.time()
    coord_transformations = None
    paths_drawer = None
    fix_paths = True
    # pdb.set_trace()
    # model = YOLO(model, device='cpu')
    model = YOLO('yolov8l-world.pt')
    video = Video(input_path=input_video)

    transformations_getter = HomographyTransformationGetter()

    motion_estimator = MotionEstimator(
        max_points=500, min_distance=7, transformations_getter=transformations_getter
    )

    distance_function = create_normalized_mean_euclidean_distance(
        video.input_height, video.input_width
    )
    distance_threshold = DISTANCE_THRESHOLD_CENTROID

    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
        past_detections_length=5,
        hit_counter_max= 5
    )

    paths_drawer = Paths(center, attenuation=0.01)

    if fix_paths:
        paths_drawer = AbsolutePaths(max_history=40, thickness=2)
    frame_count = 0
    for frame in video:
        frame_count += 1
        print("frame no:", frame_count)
        im_width, im_height = frame.shape[:2]
        if frame_count == 1 or frame_count % 2 == 0:

            yolo_out = model.predict(frame)
            mobilenet_out = detect_mobilenet(frame)
            # cv2.imwrite(f"bottle_frame{frame_count}.jpg",frame)

            # mask = np.ones(frame.shape[:2], frame.dtype)

            # coord_transformations = motion_estimator.update(frame, mask)
            # pdb.set_trace()

            yolo_detections = yolo_out_detections(yolo_out)
            mobilenet_detections = mobilenet_out_detections(mobilenet_out, im_width, im_height)
            all_detections = yolo_detections + mobilenet_detections
            all_boxes = non_max_suppression(all_detections)

            # detections2 = mobilenet_to_norfair_detections(mobilenet_detections, track_points=track_points, im_width=frame.shape[0], im_height=frame.shape[1])
            # detections = yolo_ultralytics_detections_to_norfair_detections(all_detections, track_points=track_points)
            # detections += detections2

            detections = norfair_input(all_detections, all_boxes)
            # detections = norfair_input(all_detections, nms=False)
        else:
            # mask = np.ones(frame.shape[:2], frame.dtype)
            # coord_transformations = motion_estimator.update(frame, mask)

            # detections = tracker_to_input(tracked_objects)
            detections = None


        # detections = non_max_suppression(detections)
        tracked_objects = tracker.update(
            detections=detections, #coord_transformations=coord_transformations,
            period=5
        )
        # points_detected = tracker_to_input(tracked_objects)
        print(tracked_objects)
        frame = draw(
            paths_drawer,
            track_points,
            frame,
            detections,
            tracked_objects,
            coord_transformations,
            fix_paths,
        )
        video.write(frame)

    print("time: ",time.time() - st)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument(
        "--files", type=str, default="4k_Video_20s.mp4", help="Video files to process")
    parser.add_argument(
        "--detector-path", type=str, default="yolov7.pt", help="YOLOv7 model path"
    )
    parser.add_argument(
        "--img-size", type=int, default="720", help="YOLOv7 inference size (pixels)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default="0.25",
        help="YOLOv7 object confidence threshold",
    )
    parser.add_argument(
        "--classes",
        default=0,
        nargs="+",
        type=int,
        help="Filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--track-points",
        type=str,
        default="bbox",
        help="Track points: 'centroid' or 'bbox'",
    )
    args = parser.parse_args()
    # print(args.detector_path)
    inference(
        args.files,
        args.detector_path,
        args.track_points,
        args.conf_threshold,
        args.classes,
    )
