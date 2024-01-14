# #!/usr/bin/env python3

# import argparse
# import rospy
# import cv2
# import torch
# import torch.backends.cudnn as cudnn
# import numpy as np
# from cv_bridge import CvBridge
# from pathlib import Path
# import os
# import sys
# from rostopic import get_topic_type
# import pyzed.sl as sl
# import math

# from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField, CameraInfo
# from detection_msgs.msg import BoundingBox, BoundingBoxes
# import sensor_msgs.point_cloud2 as pc2
# from std_msgs.msg import String, Float32MultiArray

# # add yolov5 submodule to path
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0] / "yolov5"
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# # import from yolov5 submodules
# from models.common import DetectMultiBackend
# from utils.general import (
#     check_img_size,
#     check_requirements,
#     non_max_suppression,
#     scale_coords,
# )
# from utils.plots import Annotator, colors
# from utils.torch_utils import select_device
# from utils.augmentations import letterbox

# @torch.no_grad()
# class Yolov5Detector:
#     def __init__(self):
#         self.conf_thres = rospy.get_param("~confidence_threshold")
#         self.iou_thres = rospy.get_param("~iou_threshold")
#         self.agnostic_nms = rospy.get_param("~agnostic_nms")
#         self.max_det = rospy.get_param("~maximum_detections")
#         self.classes = rospy.get_param("~classes", None)
#         self.line_thickness = rospy.get_param("~line_thickness")
#         self.publish_point_cloud = rospy.get_param("~publish_point_cloud")
#         self.view_image = rospy.get_param("~view_image")

#         self.x_values = []
#         self.z_values = []

#         # Initialize weights
#         weights = rospy.get_param("~weights")
#         # Initialize model
#         self.device = select_device(str(rospy.get_param("~device", "")))
#         self.model = DetectMultiBackend(
#             weights, device=self.device, dnn=rospy.get_param("~dnn"), data=rospy.get_param("~data")
#         )
#         self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
#             self.model.stride,
#             self.model.names,
#             self.model.pt,
#             self.model.jit,
#             self.model.onnx,
#             self.model.engine,
#         )

#         # Setting inference size
#         self.img_size = [rospy.get_param("~inference_size_w", 640), rospy.get_param("~inference_size_h", 480)]
#         self.img_size = check_img_size(self.img_size, s=self.stride)

#         # Half
#         self.half = rospy.get_param("~half", False)
#         self.half &= (
#             self.pt or self.jit or self.onnx or self.engine
#         ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
#         if self.pt or self.jit:
#             self.model.model.half() if self.half else self.model.model.float()
#         bs = 1  # batch_size
#         cudnn.benchmark = True  # set True to speed up constant image size inference
#         self.model.warmup()  # warmup

#         input_image_type, input_image_topic, _ = get_topic_type(
#             rospy.get_param("~input_image_topic"), blocking=True
#         )
#         self.image_sub = rospy.Subscriber(
#             rospy.get_param("~input_image_topic"), CompressedImage, self.callback, queue_size=1
#         )
#         # Initialize subscriber to depth topic
#         self.depth_sub = rospy.Subscriber(
#             rospy.get_param("~input_depth_topic"), Image, self.depth_callback, queue_size=1
#         )
#         # Camera intrinsics
#         self.camera_info_sub = rospy.Subscriber(
#             rospy.get_param("~camera_info_topic"), CameraInfo, self.camera_info_callback
#         )
#         # Initialize point_cloud publisher
#         self.pc_pub = rospy.Publisher(
#             "/yolov5/point_cloud", PointCloud2, queue_size=10
#         )
#         # Initialize prediction publisher
#         self.pred_pub = rospy.Publisher(
#             rospy.get_param("~output_topic"), BoundingBoxes, queue_size=10
#         )
#         # Initialize image publisher
#         self.publish_image = rospy.get_param("~publish_image")
#         if self.publish_image:
#             self.image_pub = rospy.Publisher(
#                 rospy.get_param("~output_image_topic"), Image, queue_size=10
#             )

#         # jyh
#         self.mj_depth_pub = rospy.Publisher(
#             "mj_depth", Float32MultiArray, queue_size=10
#         )

#         self.mj_type_pub = rospy.Publisher(
#             "mj_type", String, queue_size=10
#         )

#         # Initialize CV_Bridge
#         self.bridge = CvBridge()

#         self.distance_threshold = 1.0

#     def camera_info_callback(self, msg):
#         self.fx, self.fy, self.cx, self.cy = msg.K[0], msg.K[4], msg.K[2], msg.K[5]

#     def depth_callback(self, data):
#         self.depth = np.frombuffer(data.data, dtype=np.float32)
#         self.depth = self.depth.reshape(data.height, data.width)
#         self.depth = np.clip(self.depth, 0, 10) / 10.0
#         self.depth = np.nan_to_num(self.depth, nan=0.0)

#     def callback(self, data):
#         try:
#             # Decompress the compressed image
#             np_arr = np.frombuffer(data.data, np.uint8)
#             cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#             im, im0 = self.preprocess(cv_image)

#             if hasattr(self, "depth") and self.depth is not None:
#                 depth = self.depth.copy()
#                 depth = np.clip(depth, 0, 10) / 10.0
#             else:
#                 depth = None

#             # Run inference
#             im = torch.from_numpy(im).to(self.device)
#             im = im.half() if self.half else im.float()
#             im /= 255
#             if len(im.shape) == 3:
#                 im = im[None]

#             pred = self.model(im, augment=False, visualize=False)
#             pred = non_max_suppression(
#                 pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
#             )

#             # Process predictions
#             det = pred[0].cpu().numpy()

#             bounding_boxes = BoundingBoxes()
#             bounding_boxes.header = data.header
#             bounding_boxes.image_header = data.header

#             annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

#                 max_bbox_size = 0.0
#                 max_bbox = None
#                 max_bbox_point = None
#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     x_center = (xyxy[0] + xyxy[2]) / 2
#                     y_center = (xyxy[1] + xyxy[3]) / 2
#                     depth_val = self.depth[int(y_center), int(x_center)]

#                     if depth_val <= self.distance_threshold:
#                         bounding_box = BoundingBox()
#                         c = int(cls)
#                         # Fill in bounding box message
#                         bounding_box.Class = self.names[c]
#                         bounding_box.probability = conf
#                         bounding_box.xmin = int(xyxy[0])
#                         bounding_box.ymin = int(xyxy[1])
#                         bounding_box.xmax = int(xyxy[2])
#                         bounding_box.ymax = int(xyxy[3])
#                         bbox_size = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

#                         if bbox_size > max_bbox_size:
#                             max_bbox_size = bbox_size
#                             max_bbox = bounding_box
#                             x = (x_center - self.cx) / self.fx
#                             y = (y_center - self.cy) / self.fy
#                             z = depth_val
#                             max_bbox_point = [x, y, z]

#                         bounding_boxes.bounding_boxes.append(bounding_box)

#                         # Annotate the image
#                         if self.publish_image or self.view_image:
#                             label = f"{self.names[c]} ({x_center:.1f}, {y_center:.1f}, {depth_val:.2f})"
#                             annotator.box_label(xyxy, label, color=colors(c, True))

#                 # Create point cloud - camera inrernal parameter
#                 mirror_ref = sl.Transform()
#                 mirror_ref.set_translation(sl.Translation(0.31, -0.15, 0.22))
#                 tr_np = mirror_ref.m
#                 points = []
#                 if len(bounding_boxes.bounding_boxes) > 0:
#                     for bbox in bounding_boxes.bounding_boxes:
#                         x = bbox.xmin + (bbox.xmax - bbox.xmin) / 2
#                         y = bbox.ymin + (bbox.ymax - bbox.ymin) / 2
#                         z = bbox.z
#                         points.append([x, y, z])

#                         bbox.x = x
#                         bbox.y = y
#                         bbox.z = z

#                     # Publish point cloud
#                     if self.publish_point_cloud and max_bbox_point is not None:
#                         fields = [
#                             PointField("x", 0, PointField.FLOAT32, 1),
#                             PointField("y", 4, PointField.FLOAT32, 1),
#                             PointField("z", 8, PointField.FLOAT32, 1),
#                         ]
#                         header = data.header
#                         header.frame_id = "camera_link"
#                         transformed_points = [np.dot(tr_np, point) for point in points]
#                         transformed_points_3d = [point[:3] for point in transformed_points]

#                         pc = pc2.create_cloud(header, fields, transformed_points_3d)

#                         x_values = [point[0] for point in transformed_points_3d]
#                         z_values = [point[2] for point in transformed_points_3d]

#                         mj_topic_msg = Float32MultiArray()
#                         mj_topic_msg.data = x_values + z_values

#                         # jyh
#                         # Publish the message to mj_topic
#                         self.mj_depth_pub.publish(mj_topic_msg)
#                         self.mj_type_pub.publish(max_bbox.Class)

#                 # Stream results
#                 im0 = annotator.result()

#             # Publish prediction
#             self.pred_pub.publish(bounding_boxes)

#             # Publish & visualize images
#             if self.view_image:
#                 cv2.imshow(str(0), im0)
#                 cv2.waitKey(1)  # 1 millisecond
#             if self.publish_image:
#                 self.image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))

#         except Exception as e:
#             rospy.logerr(f"Error processing image: {str(e)}")

#     def preprocess(self, img):
#         """
#         Adapted from yolov5/utils/datasets.py LoadStreams class
#         """
#         img0 = img.copy()
#         img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
#         # Convert
#         img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
#         img = np.ascontiguousarray(img)

#         return img, img0

# if __name__ == "__main__":
#     check_requirements(exclude=("tensorboard", "thop"))

#     rospy.init_node("yolov5", anonymous=True)
#     detector = Yolov5Detector()

#     rospy.spin()





# !/usr/bin/env python3

import argparse
import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
from rostopic import get_topic_type
import pyzed.sl as sl
import math

from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField, CameraInfo
from detection_msgs.msg import BoundingBox, BoundingBoxes
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String , Float32MultiArray



# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
class Yolov5Detector:
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        self.max_det = rospy.get_param("~maximum_detections")
        self.classes = rospy.get_param("~classes", None)
        self.line_thickness = rospy.get_param("~line_thickness")
        self.publish_point_cloud = rospy.get_param("~publish_point_cloud")
        self.view_image = rospy.get_param("~view_image")
        
        self.x_values = []
        self.z_values = []
        
        # Initialize weights 
        weights = rospy.get_param("~weights")
        # Initialize model
        self.device = select_device(str(rospy.get_param("~device","")))
        self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), data=rospy.get_param("~data"))
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        # Setting inference size
        self.img_size = [rospy.get_param("~inference_size_w", 640), rospy.get_param("~inference_size_h",480)]
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half
        self.half = rospy.get_param("~half", False)
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup()  # warmup        
        
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)
        self.image_sub = rospy.Subscriber(
            rospy.get_param("~input_image_topic"), CompressedImage, self.callback, queue_size=1
        )
        # Initialize subscriber to depth topic
        self.depth_sub = rospy.Subscriber(
            rospy.get_param("~input_depth_topic"), Image, self.depth_callback, queue_size = 1,
        )
        # Camera intrinsics
        self.camera_info_sub = rospy.Subscriber(
            rospy.get_param("~camera_info_topic"), CameraInfo, self.camera_info_callback
        )
        # Initialize point_cloud publisher
        self.pc_pub = rospy.Publisher(
            "/yolov5/point_cloud", PointCloud2, queue_size=10
        )
        # Initialize prediction publisher
        self.pred_pub = rospy.Publisher(
            rospy.get_param("~output_topic"), BoundingBoxes, queue_size=10
        )
        # Initialize image publisher
        self.publish_image = rospy.get_param("~publish_image")
        if self.publish_image:
            self.image_pub = rospy.Publisher(
                rospy.get_param("~output_image_topic"), Image, queue_size=10
            )

        #jyh
        self.mj_depth_pub = rospy.Publisher(
            "mj_depth", Float32MultiArray, queue_size=10
        )

        self.mj_type_pub = rospy.Publisher(
            "mj_type", String, queue_size=10
        )

        # Initialize CV_Bridge
        self.bridge = CvBridge()

        self.distance_threshold = 1.0

    def camera_info_callback(self, msg):
        self.fx, self.fy, self.cx, self.cy = msg.K[0], msg.K[4], msg.K[2], msg.K[5]

    def depth_callback(self, data):
        self.depth = np.frombuffer(data.data, dtype=np.float32)
        self.depth = self.depth.reshape(data.height, data.width)
        self.depth = np.clip(self.depth, 0, 10) / 10.0
        self.depth = np.nan_to_num(self.depth, nan=0.0)
    
    def callback(self, data):
        """adapted from yolov5/detect.py"""
        # cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        np_arr = np.frombuffer(data.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        
        im, im0 = self.preprocess(cv_image)

        if hasattr(self, "depth") and self.depth is not None:
            depth = self.depth.copy()
            depth = np.clip(depth, 0, 10) / 10.0
        else:
            depth = None

        # Run inference
        im = torch.from_numpy(im).to(self.device) 
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )

        ### To-do move pred to CPU and fill BoundingBox messages
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = data.header
        bounding_boxes.image_header = data.header
        
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names)) 

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            max_bbox_size = 0.0
            max_bbox = None
            max_bbox_point = None
            # Write results
            for *xyxy, conf, cls in reversed(det):
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2
                depth_val = self.depth[int(y_center), int(x_center)]
                # print("Object_D:", x_center, y_center, depth_val)

                if depth_val <= self.distance_threshold:
                    bounding_box = BoundingBox()
                    c = int(cls)
                    # Fill in bounding box message
                    bounding_box.Class = self.names[c]   ############


                    bounding_box.probability = conf 
                    bounding_box.xmin = int(xyxy[0])
                    bounding_box.ymin = int(xyxy[1])
                    bounding_box.xmax = int(xyxy[2])
                    bounding_box.ymax = int(xyxy[3])
                    bbox_size = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    # class_msg = Class()
                    # class_msg.Class = self.names[c]

                    
                    if bbox_size > max_bbox_size:
                        max_bbox_size = bbox_size 
                        max_bbox = bounding_box
                        x = (x_center - self.cx) / self.fx  
                        y = (y_center - self.cy) / self.fy
                        z = depth_val
                        # z = self.depth[int(y_center), int(x_center)]
                        max_bbox_point = [x, y, z]

                    bounding_boxes.bounding_boxes.append(bounding_box)

                    # Annotate the image
                    if self.publish_image or self.view_image:  # Add bbox to image
                          # integer class
                        label = f"{self.names[c]} ({x_center:.1f}, {y_center:.1f}, {depth_val:.2f})"
                        annotator.box_label(xyxy, label, color=colors(c, True))       

            # Create point cloud - camera inrernal parameter
            mirror_ref = sl.Transform()
            mirror_ref.set_translation(sl.Translation(0.31,-0.15,0.22))
            tr_np = mirror_ref.m
            points = []
            if len(bounding_boxes.bounding_boxes) > 0:            
                for bbox in bounding_boxes.bounding_boxes:
                    x = (x_center - self.cx) / self.fx  
                    y = (y_center - self.cy) / self.fy
                    z = depth_val
                    # z = self.depth[int(y_center), int(x_center)]
                    max_bbox_point = [x, y, z]
                    points.append(max_bbox_point)
                    # print("Object_P:", point)

                    bbox.x = x
                    bbox.y = y
                    bbox.z = z

                # Publish point cloud
                if self.publish_point_cloud and max_bbox_point is not None:
                    fields = [
                        PointField("x", 0, PointField.FLOAT32, 1),
                        PointField("y", 4, PointField.FLOAT32, 1),
                        PointField("z", 8, PointField.FLOAT32, 1),
                    ]
                    header = data.header
                    header.frame_id = "camera_link"
                    # point_cloud_np = np.array(points)
                    # transformed_point_cloud = point_cloud_np.dot(tr_np)
                    points_homogeneous = [max_bbox_point + [1]]  # make points homogeneous by appending 1 to each point
                    # print("Object_H:", points_homogeneous)                    
                    transformed_points = [np.dot(tr_np, point) for point in points_homogeneous]  # apply transformation
                    transformed_points_3d = [point[:3] for point in transformed_points]
                    
                    pc = pc2.create_cloud(header, fields, transformed_points_3d)
                    print("Object_T:", transformed_points_3d)
                    print(bounding_box.Class)
                    x_values = [point[0] for point in transformed_points_3d]
                    z_values = [point[2] for point in transformed_points_3d]


                    mj_topic_msg = Float32MultiArray()
                    mj_topic_msg.data = x_values + z_values

                    #jyh
                    # Publish the message to mj_topic
                    self.mj_depth_pub.publish(mj_topic_msg)
                    self.mj_type_pub.publish(bounding_box.Class)
                    # Print 'x' and 'z' values



                ### POPULATE THE DETECTION MESSAGE HERE

            # Stream results
            im0 = annotator.result()

        # Publish prediction
        self.pred_pub.publish(bounding_boxes)
        
        # self.class_pub.publish(class_msg)

        # Publish & visualize images
        if self.view_image:
            cv2.imshow(str(0), im0)
            cv2.waitKey(1)  # 1 millisecond
        if self.publish_image:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))
        

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0 


if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop"))
    

    rospy.init_node("yolov5", anonymous=True)
    detector = Yolov5Detector()
    
    rospy.spin()