"""
This module provides functionality to annotate images using the YOLOv8 model. It offers the YoloAnnotator
class, which can detect objects in an image and uses the SAM model for refining the object hypotheses RoIs
when precision mode is active.

Author: Lennart Heinbokel & Sorin Arion
"""
import copy
import json
import os

import cv2
import numpy as np
import open3d as o3d
import py_trees
import robokudo.annotators
import robokudo.types.scene
import robokudo.utils.annotator_helper
import robokudo.utils.cv_helper
from robokudo.cas import CASViews
from robokudo.utils.decorators import timer_decorator
from ultralytics import YOLO, SAM
import torch


class YoloAnnotator(robokudo.annotators.core.ThreadedAnnotator):
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            """
            This class contains all parameters that are necessary for the YoloAnnotator.

            Attributes:
                ros_pkg_path:       The name of the ROS package that contains the weights and id2name file.
                weights_path:       The relative path to the weights file, starting from the ROS package.
                pretrained_model:   Use a pretrained model of YOLO instead of providing weights on your filesystem.
                sam_weights_path:   The path to the ViT checkpoint, needed by SAM. Required if precision_mode is set to True.
                sam_model_type:     The type of the ViT backbone. Choose either 'vit_b' or 'vit_h'. Required if precision_mode is set to True.
                id2name_json_path:  The relative path to the id2name json file, starting from the ROS package.
                threshold:          The threshold for the confidence score of the YOLOv8 model.
                                    Detections with a lower score are ignored.
                precision_mode:     If set to True, SAM will be used to refine the object hypotheses ROIs.
            """

            def __init__(self):
                self.ros_pkg_path = None
                self.weights_path = None  # TODO: git lfs weights and set default param here
                self.pretrained_model = None  #
                self.sam_model = "mobile_sam.pt"

                self.id2name_json_path = "data/json/id2name.json"
                self.threshold = 0.9
                self.precision_mode = False

        parameters = Parameters()

    def __init__(self, name="YoloAnnotator", descriptor=Descriptor()):
        """Initializes the YoloAnnotator.

        Args:
            name:       The name of the annotator.
            descriptor: The YoloAnnotator.Descriptor object that contains the parameters.
        """
        super(YoloAnnotator, self).__init__(name, descriptor)

        if self.descriptor.parameters.pretrained_model:
            self.model = YOLO(self.descriptor.parameters.pretrained_model)
        else:
            self.model = YOLO(self._get_weight_path())
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.descriptor = descriptor

        if self.descriptor.parameters.pretrained_model is not None:
            self.id2name = self.model.names
        else:
            with open(self._get_json_path(), "r", encoding="utf-8") as file:
                self.id2name = json.load(file)

        if descriptor.parameters.precision_mode:
            self.sam = SAM(self.descriptor.parameters.sam_model)

        self.cam_intrinsics = None

    def _get_path(self, path_type) -> str:
        """Returns the full path to the either the weights or the id2name json file."""
        path_map = {
            "json": self.descriptor.parameters.id2name_json_path,
            "weight": self.descriptor.parameters.weights_path,
        }

        if self.descriptor.parameters.ros_pkg_path is None or path_map[path_type] is None:
            raise ValueError(
                f"Invalid parameters for {path_type}: {path_map[path_type]}. Please refer to the Parameter class's docstring."
            )

        file_loader = robokudo.utils.file_loader.FileLoader()
        full_path = file_loader.get_path_to_file_in_ros_package(
            ros_pkg_name=self.descriptor.parameters.ros_pkg_path,
            relative_path=path_map[path_type],
        )

        return str(full_path)

    def _get_json_path(self) -> str:
        return self._get_path("json")

    def _get_weight_path(self) -> str:
        return self._get_path("weight")

    def load_image(self):
        """Load and resize the color image from the CAS.

        This is useful, as the resolution of the RGB image might
        differ from the resolution of the depth image. This function
        resizes the color image to the resolution of the depth image.

        This is important later on when we want to create a point cloud
        from the depth image and the color image and crop the point cloud
        to the bounding box of the object hypothesis, which is defined in
        the color image coordinate system.

        Returns:
            The resized color image.

        Raises:
            RuntimeError: If no color to depth ratio is provided by the camera driver.
        """
        img = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        resized_color = None

        if self.descriptor.parameters.global_with_depth:
            try:
                resized_color = robokudo.utils.cv_helper.get_scaled_color_image_for_depth_image(self.get_cas(), img)
            except RuntimeError as e:  # pylint: disable=invalid-name
                self.rk_logger.error(f"No color to depth ratio set by your camera driver! Can't preprocess: {e}")
        else:
            resized_color = img

        return resized_color

    def resize_mask(self, mask):
        """
        The mask is potentially created after the input image has been scaled down. If that's the case,
        we have to bring it back to the original resolution.

        :param mask: A binary image.
        :return: The scaled version of mask, according to the COLOR2DEPTH_RATIO
        """
        if not self.descriptor.parameters.global_with_depth:
            # nothing to do
            return mask

        color2depth_ratio = self.get_cas().get(robokudo.cas.CASViews.COLOR2DEPTH_RATIO)

        if not color2depth_ratio:
            raise RuntimeError("No Color to Depth Ratio set. Can't continue.")

        if color2depth_ratio == (1, 1):
            return mask
        else:
            c2d_ratio_x = color2depth_ratio[0]
            c2d_ratio_y = color2depth_ratio[1]
            resized_mask = cv2.resize(mask, None,
                                      fx=1/c2d_ratio_x,
                                      fy=1/c2d_ratio_y, interpolation=cv2.INTER_NEAREST)

        return resized_mask

    # pylint: disable=too-many-locals, too-many-statements
    @timer_decorator
    def compute(self):
        """Computes the object hypotheses and adds them to the CAS."""
        img = self.load_image()

        if self.descriptor.parameters.global_with_depth:
            self.cam_intrinsics = copy.deepcopy(self.get_cas().get(CASViews.CAM_INTRINSIC))
            robokudo.utils.annotator_helper.scale_cam_intrinsics(self)

            depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
            color2depth_ratio = self.get_cas().get(CASViews.COLOR2DEPTH_RATIO)

        with torch.no_grad():
            results = self.model.predict(img, conf=self.descriptor.parameters.threshold, show=False)

        object_hypotheses = []
        for result in results:
            for box in result.boxes:
                results_numpy = box.cpu().numpy()
                bbox = box.xyxy.tolist()[0]
                # pylint: disable=invalid-name
                x1, y1, x2, y2 = (results_numpy.xyxy[0][0],
                                  results_numpy.xyxy[0][1],
                                  results_numpy.xyxy[0][2],
                                  results_numpy.xyxy[0][3])

                if self.descriptor.parameters.precision_mode:
                    masks = self.sam.predict(img, bboxes=[bbox], labels=[1])[0].masks.data.cpu().numpy()
                    mask = masks[0].astype(np.uint8)
                else:
                    if self.descriptor.parameters.global_with_depth:
                        mask = np.zeros_like(depth, dtype=np.uint8)
                        mask[int(y1): int(y2), int(x1): int(x2)] = 1
                    else:
                        mask = np.ones(img.shape[:2], dtype=np.uint8)

                torch.cuda.empty_cache()

                if self.descriptor.parameters.global_with_depth:
                    # Respect the color2depth ratio, when creating the cluster cloud
                    depth2color_ratio = (1 / color2depth_ratio[0], 1 / color2depth_ratio[1])
                    x1 = (x1 * depth2color_ratio[0]).astype(int)
                    x2 = (x2 * depth2color_ratio[0]).astype(int)
                    y1 = (y1 * depth2color_ratio[1]).astype(int)
                    y2 = (y2 * depth2color_ratio[1]).astype(int)

                    depth_masked = copy.deepcopy(depth)
                    depth_masked = np.where(mask == 1, depth_masked, 0)

                    o3d_color = o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    o3d_depth = o3d.geometry.Image(depth_masked)
                    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth,
                                                                                    convert_rgb_to_intensity=False)
                    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd_image, copy.deepcopy(self.get_cas().get(CASViews.CAM_INTRINSIC))
                    )

                    self.get_annotator_output_struct().set_geometries(pcd)
                else:
                    x1 = x1.astype(int)
                    y1 = y1.astype(int)
                    x2 = x2.astype(int)
                    y2 = y2.astype(int)
                    pcd = None

                name = results_numpy.cls[0]

                if self.descriptor.parameters.pretrained_model:
                    # Pretrained model id2name uses INT as keys.
                    classname = self.id2name.get(int(name), "unknown. ID: " + str(int(name)))
                else:
                    # TODO remove when everything is migrated to the standard id2name mapping of the YOLO() object
                    # id2name json file uses STR as keys
                    classname = self.id2name.get(str(int(name)), "unknown. ID: " + str(int(name)))

                classification = robokudo.types.annotation.Classification()
                classification.classname = classname
                classification.confidence = results_numpy.conf[0]

                if classname == "person":
                    # detect human
                    object_hypothesis = robokudo.types.scene.HumanHypothesis()
                else:
                    # detect some object
                    object_hypothesis = robokudo.types.scene.ObjectHypothesis()
                object_hypothesis.type = name
                object_hypothesis.points = pcd
                w = x2 - x1  # pylint: disable=invalid-name
                h = y2 - y1  # pylint: disable=invalid-name

                #object_hypothesis.bbox = [x1, y1, x2, y2]  # against class definition
                object_hypothesis.roi.roi.pos.x = x1
                object_hypothesis.roi.roi.pos.y = y1
                object_hypothesis.roi.roi.width = w
                object_hypothesis.roi.roi.height = h

                # Crop mask to ROI
                # The standard convention for masks is to be constrained to the ROI they are relative to
                # This allows following annotators to iterate fast over the mask
                object_hypothesis.roi.mask = self.resize_mask(mask)     # scale mask back to original if necessary
                object_hypothesis.roi.mask *= 255   # SAM outputs masks with 1. Scale to 255.
                object_hypothesis.roi.mask = \
                    robokudo.utils.cv_helper.crop_image(object_hypothesis.roi.mask, (x1, y1), (w, h))

                #object_hypothesis.classification = classification  # against class definition
                object_hypothesis.annotations.append(classification)
                object_hypotheses.append(object_hypothesis)

        self.get_cas().annotations.extend(object_hypotheses)

        if self.descriptor.parameters.global_with_visualization:
            if self.descriptor.parameters.precision_mode:
                self.vis_precision_mode(object_hypotheses)
            else:
                self.vis_base_mode(object_hypotheses)

        return py_trees.Status.SUCCESS

    def vis_base_mode(self, object_hypotheses):
        """Visualizes the object hypotheses in the base mode."""
        visualization_img = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        vis_geometries = []

        for oh in object_hypotheses:  # pylint: disable=invalid-name
            assert isinstance(oh, robokudo.types.scene.ObjectHypothesis)
            # pylint: disable=invalid-name
            x1, y1, x2, y2 = (oh.roi.roi.pos.x,
                              oh.roi.roi.pos.y,
                              oh.roi.roi.pos.x + oh.roi.roi.width,
                              oh.roi.roi.pos.y + oh.roi.roi.height)

            text = f"{oh.annotations[0].classname}, {oh.annotations[0].confidence:.2f}"
            font = cv2.FONT_HERSHEY_COMPLEX
            visualization_img = cv2.putText(visualization_img, text, (x1, y1-5), font, 0.5,
                                            (0, 0, 255), 1, 2)
            visualization_img = cv2.rectangle(visualization_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            vis_geometries.append(oh.points)

        self.get_annotator_output_struct().set_image(visualization_img)
        if self.descriptor.parameters.global_with_depth:
            self.get_annotator_output_struct().set_geometries(vis_geometries)

    def vis_precision_mode(self, object_hypotheses):
        """Visualizes the object hypotheses in the precision mode."""
        visualization_img = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        cmap = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [122, 0, 122], [90, 90, 0]])
        vis_geometries = []

        for oh_idx, oh in enumerate(object_hypotheses):  # pylint: disable=invalid-name
            assert isinstance(oh, robokudo.types.scene.ObjectHypothesis)
            # pylint: disable=invalid-name
            x1, y1, x2, y2 = (oh.roi.roi.pos.x,
                              oh.roi.roi.pos.y,
                              oh.roi.roi.pos.x + oh.roi.roi.width,
                              oh.roi.roi.pos.y + oh.roi.roi.height)

            vis_text = f"{oh.annotations[0].classname}, {oh.annotations[0].confidence:.2f}"
            font = cv2.FONT_HERSHEY_COMPLEX
            visualization_img = cv2.putText(visualization_img, vis_text, (x1, y1-5), font, 0.5,
                                            (0, 0, 255), 1, 2)
            visualization_img = cv2.rectangle(visualization_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            full_mask = np.zeros((visualization_img.shape[0], visualization_img.shape[1]), dtype="uint8")
            full_mask[y1: y1+oh.roi.mask.shape[0],
                      x1: x1+oh.roi.mask.shape[1]] = oh.roi.mask

            visualization_img[full_mask == 255] = cmap[oh_idx % cmap.shape[0]]
            vis_geometries.append(oh.points)

        self.get_annotator_output_struct().set_image(visualization_img)
        if self.descriptor.parameters.global_with_depth:
            self.get_annotator_output_struct().set_geometries(vis_geometries)
