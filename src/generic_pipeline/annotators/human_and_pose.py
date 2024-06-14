import copy
from dataclasses import dataclass
from timeit import default_timer

import cv2
import numpy
import numpy as np
import open3d as o3d
import py_trees
from ultralytics import YOLO

import robokudo.annotators
from robokudo.cas import CASViews
from robokudo.types.annotation import CloudAnnotation
from robokudo.types.human import KeypointAnnotation
from robokudo.types.scene import HumanHypothesis
from robokudo.utils import cv_helper
from robokudo.utils import o3d_helper
import robokudo.utils.decorators


class HumanAndPoseAnnotator(robokudo.annotators.core.BaseAnnotator):
    @dataclass
    class GetKeypoint:
        NOSE: int = 0
        LEFT_EYE: int = 1
        RIGHT_EYE: int = 2
        LEFT_EAR: int = 3
        RIGHT_EAR: int = 4
        LEFT_SHOULDER: int = 5
        RIGHT_SHOULDER: int = 6
        LEFT_ELBOW: int = 7
        RIGHT_ELBOW: int = 8
        LEFT_WRIST: int = 9
        RIGHT_WRIST: int = 10
        LEFT_HIP: int = 11
        RIGHT_HIP: int = 12
        LEFT_KNEE: int = 13
        RIGHT_KNEE: int = 14
        LEFT_ANKLE: int = 15
        RIGHT_ANKLE: int = 16

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.labels = []
                self.yolo_model_name = 'yolov8n-pose.pt'
                self.ros_pkg = 'robokudo_human_detection'
                self.model_path = 'models'

                # Minimum confidence to accept a human detection, otherwise ignore
                self.confidence_threshold = 0.7

                # Set to a float [0...1] if Boundingbox of detected human should be shrinked
                self.shrink_bounding_box = None

                # If the Bounding Box shall be shrinked, decide if the shrinked bbox should be placed directly on the
                # Human Hypothesis, or if the shrinked version should only be added as a Annotation to the HumanHyp
                self.shrink_as_annotation = False

                #### Parameters devoted to the creation of keypoint clouds
                # Set to true to activate. Warning: This is computationally expensive.
                self.generate_clouds_for_keypoints = False
                # Only generate clouds when keypoint confidence is over:
                self.keypoint_confidence_threshold_for_clouds = 0.6
                # The pixel radius around each keypoint to incorporate into the cloud generation
                self.keypoint_masking_radius = 6

                self.warmup_annotator = True

                # Ignore depth measurements above this value in meters.
                # This removes potentially unnecessary background from your human detection.
                self.depth_truncate = 3.5

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="HumanAndPoseAnnotator", descriptor=Descriptor()):
        super(HumanAndPoseAnnotator, self).__init__(name, descriptor)
        self.yolo_model = None
        self.text = self.descriptor.parameters.labels
        self.setup(10)
        self.cam_intrinsics = None

    def setup(self, timeout):
        file_loader = robokudo.utils.file_loader.FileLoader()
        model_folder_path = file_loader.get_path_to_file_in_ros_package(
            ros_pkg_name=self.descriptor.parameters.ros_pkg,
            relative_path=self.descriptor.parameters.model_path)
        self.yolo_model = YOLO(model_folder_path / self.descriptor.parameters.yolo_model_name)

        if self.descriptor.parameters.warmup_annotator:
            empty_image = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
            warmup_inference = self.yolo_model(empty_image, save=False, verbose=False, iou=0.5)

        return True

    @robokudo.utils.decorators.record_time
    def update(self):
        start_timer = default_timer()
        color = self.get_cas().get(CASViews.COLOR_IMAGE)

        if self.descriptor.parameters.global_with_depth:
            # for pointcloud generation
            depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
            self.cam_intrinsics = copy.deepcopy(self.get_cas().get(CASViews.CAM_INTRINSIC))

            color2depth_ratio = self.get_cas().get(robokudo.cas.CASViews.COLOR2DEPTH_RATIO)

            # Scale the image down so that it matches the depth image size
            try:
                resized_color = robokudo.utils.cv_helper.get_scaled_color_image_for_depth_image(self.get_cas(), color)
                robokudo.utils.annotator_helper.scale_cam_intrinsics(self)
            except RuntimeError as e:
                self.rk_logger.error(
                    "No color to depth ratio set by your camera driver! Can't scale image for Point Cloud creation.")
                raise Exception(
                    "No color to depth ratio set by your camera driver! Can't scale image for Point Cloud creation.")
        else:
            resized_color = color

        idx = 0

        results = self.yolo_model(color, save=False, verbose=False, iou=0.5,
                                  conf=self.descriptor.parameters.confidence_threshold)

        all_vis_clouds = []
        for i in range(len(results[0].boxes.data)):
            box_dim = results[0].boxes.data[i].squeeze().tolist()

            # Shrink the bounding box - This might interfere with Activity Recognition!
            if self.descriptor.parameters.shrink_bounding_box is not None:
                shrinked_box_dim = copy.deepcopy(box_dim)
                shrink_factor = self.descriptor.parameters.shrink_bounding_box
                shrink_factor_half = shrink_factor / 2
                shrinked_box_x_len, shrinked_box_y_len = self.get_box_xy_dim(shrinked_box_dim)

                shrinked_box_dim[0] += shrinked_box_x_len * shrink_factor_half
                shrinked_box_dim[2] -= shrinked_box_x_len * shrink_factor_half
                shrinked_box_dim[1] += shrinked_box_y_len * shrink_factor_half
                shrinked_box_dim[3] -= shrinked_box_y_len * shrink_factor_half

                if self.descriptor.parameters.global_with_depth and self.descriptor.parameters.shrink_as_annotation:
                    shrinked_cloud, _ = self.get_cloud_for_bbox(shrinked_box_dim, color2depth_ratio, depth,
                                                                resized_color, self.cam_intrinsics,
                                                                self.descriptor.parameters.depth_truncate)
                    # Set normal dimensions for bounding box in HumanHyp in this case
                    box_x_len, box_y_len = self.get_box_xy_dim(box_dim)
                else:
                    # Overwrite bounding box dimensions with shrinked version
                    box_x_len, box_y_len = self.get_box_xy_dim(shrinked_box_dim)
                    box_dim = shrinked_box_dim
            else:
                box_x_len, box_y_len = self.get_box_xy_dim(box_dim)

            if self.descriptor.parameters.global_with_depth:
                # create a pointcloud for the human BB
                cloud, color_rgb = self.get_cloud_for_bbox(box_dim, color2depth_ratio, depth, resized_color,
                                                           self.cam_intrinsics, self.descriptor.parameters.depth_truncate)
            else:
                cloud = None

            # EXPERIMENTAL
            # Cluster for the objects big enough and front most
            # Takes an extra 10-30ms, depending on your parameters
            #
            # with o3d.utility.VerbosityContextManager(
            #         o3d.utility.VerbosityLevel.Debug) as cm:
            #     labels = np.array(
            #         cloud.cluster_dbscan(eps=0.03, min_points=100, print_progress=True))
            #
            # max_label = labels.max()
            # print(f"point cloud has {max_label + 1} clusters")

            # if this fails on your machine, update ultralytics to something >= 8.0.123
            human_keypoints2d = results[0].keypoints[i].data.cpu().numpy().squeeze().tolist()

            hh = HumanHypothesis()
            hh.id = idx
            idx += 1
            hh.source = self.name
            hh.points = cloud
            hh.roi.roi.pos.x = int(box_dim[0])
            hh.roi.roi.pos.y = int(box_dim[1])
            hh.roi.roi.width = int(box_x_len)
            hh.roi.roi.height = int(box_y_len)

            # Store the confidence as classification
            classification = robokudo.types.annotation.Classification()
            classification.classname = "person"
            classification.confidence = results[0].boxes.conf[i].cpu().numpy()
            hh.annotations.append(classification)

            if self.descriptor.parameters.global_with_depth and self.descriptor.parameters.shrink_as_annotation:
                ca = CloudAnnotation()
                ca.source = type(self).__name__
                ca.points = shrinked_cloud

                hh.annotations.append(ca)

            human_keypoints = KeypointAnnotation()
            human_keypoints.keypoints = human_keypoints2d  # list of 3-elem list. x/y/confidence
            human_keypoints.type = KeypointAnnotation.KP_TYPE_2D
            hh.annotations.append(human_keypoints)

            if self.descriptor.parameters.global_with_depth and self.descriptor.parameters.generate_clouds_for_keypoints:
                human_keypoints3d = KeypointAnnotation()
                human_keypoints3d.type = KeypointAnnotation.KP_TYPE_3D

                human_keypoints3d_list = []
                for human_keypoint in human_keypoints2d:
                    x, y, confidence = human_keypoint

                    if confidence < self.descriptor.parameters.keypoint_confidence_threshold_for_clouds:
                        # Skip cloud generation if confidence is too low
                        # Generate an empty cloud
                        human_keypoints3d_list.append([o3d.geometry.PointCloud(), confidence])
                        continue

                    # Scale x and y to match depth resolution if necessary
                    x = int(x * color2depth_ratio[0])
                    y = int(y * color2depth_ratio[1])

                    kp_depth_mask = np.zeros_like(depth, dtype=numpy.uint8)
                    # kp_depth_mask = cv_helper.draw_rectangle_around_center(kp_depth_mask, x, y, 20, 20)
                    kp_depth_mask = cv2.circle(kp_depth_mask, (x, y),
                                               self.descriptor.parameters.keypoint_masking_radius, 255,
                                               thickness=cv2.FILLED)

                    depth_copy = copy.deepcopy(depth)
                    kp_cloud = o3d_helper.get_cloud_from_rgb_depth_and_mask(rgb_image=color_rgb, depth_image=depth_copy,
                                                                            mask=kp_depth_mask,
                                                                            cam_intrinsics=self.cam_intrinsics)

                    human_keypoints3d_list.append([kp_cloud, confidence])
                    # all_vis_clouds.append(kp_cloud)

                human_keypoints3d.keypoints = human_keypoints3d_list
                hh.annotations.append(human_keypoints3d)

            all_vis_clouds.append(cloud)

            self.get_cas().annotations.append(hh)

        if self.descriptor.parameters.global_with_visualization:
            annotated_frame = results[0].plot()
            self.get_annotator_output_struct().set_image(annotated_frame)

            if self.descriptor.parameters.global_with_depth and len(all_vis_clouds) > 0:
                visualization_cloud = robokudo.utils.o3d_helper.concatenate_clouds(all_vis_clouds)
                self.get_annotator_output_struct().set_geometries(visualization_cloud)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS

    def get_box_xy_dim(self, box_dim):
        box_x_len = box_dim[2] - box_dim[0]
        box_y_len = box_dim[3] - box_dim[1]
        return box_x_len, box_y_len

    def get_cloud_for_bbox(self, box_dim, color2depth_ratio, depth, resized_color, cam_intrinsics, depth_truncate=3.5):
        x1 = int(box_dim[0] * color2depth_ratio[0])
        y1 = int(box_dim[1] * color2depth_ratio[1])
        x2 = int(box_dim[2] * color2depth_ratio[0])
        y2 = int(box_dim[3] * color2depth_ratio[1])
        bb_mask = np.zeros_like(depth, dtype=np.uint8)
        cv2.rectangle(bb_mask, (x1, y1), (x2, y2), 255, -1)
        color_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
        depth_copy = copy.deepcopy(depth)
        cloud = o3d_helper.get_cloud_from_rgb_depth_and_mask(rgb_image=color_rgb, depth_image=depth_copy,
                                                             mask=bb_mask,
                                                             cam_intrinsics=cam_intrinsics,
                                                             depth_truncate=depth_truncate)
        return cloud, color_rgb
