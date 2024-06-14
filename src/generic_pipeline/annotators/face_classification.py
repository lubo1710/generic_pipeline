import copy
import os
from timeit import default_timer

import face_recognition
import py_trees
from pathlib import Path

import robokudo
import robokudo.annotators.core
import robokudo.utils.cv_helper
import robokudo.utils.error_handling
import robokudo.types.annotation
import robokudo.types.scene
from robokudo.cas import CASViews


class FaceClassification(robokudo.annotators.core.BaseAnnotator):
    """Classifies detected faces based on face_recognition library."""

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.ros_pkg_path = None  # If set, use use data_path as a relative path to self.ros_pkg_path
                self.data_path = None  # Relative Path to the folder containing the models
                self.file_names = []  # files in self.data_path to load
                self.labels = []  # 'class labels' for each of the file
                self.confidence = 0.6

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="FaceClassification", descriptor=Descriptor()):
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.known_face_encodings = []
        self.load_faces()

    def load_faces(self):
        if self.descriptor.parameters.ros_pkg_path is None or self.descriptor.parameters.data_path is None:
            raise RuntimeError(f"Can't look up necessary input data. Please check Parameters of this Annotator.")

        file_loader = robokudo.utils.file_loader.FileLoader()
        data_folder_path = file_loader.get_path_to_file_in_ros_package(
            ros_pkg_name=self.descriptor.parameters.ros_pkg_path,
            relative_path=self.descriptor.parameters.data_path)
        if len(self.descriptor.parameters.file_names) != len(self.descriptor.parameters.labels):
            raise RuntimeError("File names and corresponding label list must be of equal size")

        # Append self.descriptor.parameters.file_names
        filenames = []
        classes = []
        for filename in os.listdir(data_folder_path):
            filenames.append(filename)
            classe = filename.split(".")[0]
            classes.append(classe)
        self.descriptor.parameters.file_names = filenames  # files in self.data_path to load
        self.descriptor.parameters.labels = classes  # 'class labels' for each of the fil

        print(f' Set descriptor to: {filenames} and {classes}')

        for file_name in self.descriptor.parameters.file_names:
            file_path: Path = data_folder_path.joinpath(f"{file_name}")
            if not file_path.exists():
                self.rk_logger.error(f"No face image file found at '{file_path}'")

            known_person_image = face_recognition.load_image_file(str(file_path))
            known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
            self.known_face_encodings.append(known_person_encoding)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self):
        self.load_faces() # Reload faces before comparison

        start_timer = default_timer()

        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)

        human_hypotheses = self.get_cas().filter_annotations_by_type(robokudo.types.scene.HumanHypothesis)

        if len(human_hypotheses) == 0:
            self.rk_logger.warning(
                f"Could not find any HumanHypothesis")
            return py_trees.Status.SUCCESS # raise Exception("No humans detected - Can't classify them")

        for human_hypothesis in human_hypotheses:
            encoding_annotations = robokudo.cas.CAS.filter_by_type(robokudo.types.annotation.Encoding,
                                                                   human_hypothesis.annotations)
            for encoding_annotation in encoding_annotations:
                if encoding_annotation.source != 'FaceDetector':
                    continue

                # Classify the given face by comparing against the loaded face encodings
                results = face_recognition.compare_faces(self.known_face_encodings,
                                                         encoding_annotation.encoding[0])
                detected_classes = [name for (match, name) in zip(results, self.descriptor.parameters.labels) if match]

                # TODO: Make majority voting scheme or switch to kNN
                if len(detected_classes) > 0:
                    detected_class = detected_classes[0]
                    classification = robokudo.types.annotation.Classification()
                    classification.source = self.name
                    classification.classification_type = 'face'
                    classification.classname = detected_class
                    human_hypothesis.annotations.append(classification)
                break

        visualization_img = copy.deepcopy(self.color)
        self.draw_visualization(visualization_img)
        self.get_annotator_output_struct().set_image(visualization_img)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS

    def draw_visualization(self, visualization_img):
        """
        Fetch object hypothesis with Annotations done by this Annotator and visualize them in 2D
        :param visualization_img:
        :return: The updated visualization_img with the bounding boxes and class information
        """
        human_hypotheses = self.get_cas().filter_annotations_by_type(robokudo.types.scene.HumanHypothesis)
        human_hypotheses_with_face_encodings = []

        for human_hypothesis in human_hypotheses:
            encoding_annotations = robokudo.cas.CAS.filter_by_type(robokudo.types.annotation.Encoding,
                                                                   human_hypothesis.annotations)
            for encoding_annotation in encoding_annotations:
                if encoding_annotation.source != 'FaceDetector':
                    continue

                human_hypotheses_with_face_encodings.append(human_hypothesis)
                # only add one hypothesis with an encoding
                break

        # Fetch the class annotations done by this annotator
        # Warning:If you have multiple Annotators of this type in your BT, you need to switch from self.get_class_name()
        # in the annotation to self.name and set different names in the CTOR. We also assume,
        # that no annotations from this Annotator have existed in the CAS prior to the call of this Annotator
        def get_class_label_from_h(h):
            class_annotations = [a for a in h.annotations \
                                 if isinstance(a, robokudo.types.annotation.Classification)
                                 and a.source == self.name]

            if len(class_annotations) == 1:
                return class_annotations[0].classname
            return "NO-CLASS"

        robokudo.utils.annotator_helper. \
            draw_bounding_boxes_from_object_hypotheses(visualization_img,
                                                       human_hypotheses_with_face_encodings,
                                                       lambda h: f"{get_class_label_from_h(h)}")

        return visualization_img
