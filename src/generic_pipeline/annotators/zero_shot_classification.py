"""
This module implements a RoboKudo Annotator: The ZeroShotClfAnnotator.
This annotator uses a pretrained CLIP model to classify the objects in the scene, 
in a zero-shot manner.

Author: Lennart Heinbokel
Created at: 2023-06-01
"""
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import typing
from timeit import default_timer

import PIL
import numpy as np
import py_trees
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import robokudo.annotators
import robokudo.types
import robokudo.types.scene
from robokudo.cas import CASViews
from robokudo.utils.error_handling import catch_and_raise_to_blackboard


class ZeroShotClfAnnotator(robokudo.annotators.core.ThreadedAnnotator):
    # pylint: disable=too-few-public-methods
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        """Descriptor for the ZeroShotClfAnnotator."""

        # TODO: write documentation for inputs and outputs for this annotator...

        # these can be used as a template for the classification
        example_classes = {
            "human_activities": [
                "Walking",
                "Running",
                "Sitting",
                "Lying",
                "Crouching",
                "Waving",
            ],
            "fruits": [
                "Apple",
                "Banana",
                "Orange",
                "Strawberry",
                "Plum",
                "Peach",
                "Pear",
                "Pineapple",
            ],
            "balls": [
                "Basketball",
                "Baseball",
                "Football",
                "Softball",
                "Tennis ball",
                "Softball",
                "Volleyball",
                "Soccer ball",
            ],
        }

        class Parameters:
            """Parameters for the Descriptor."""

            # pylint: disable=too-many-arguments
            def __init__(self):
                """Initialize the parameters.

                Args:
                    classes:        List of classes to classify. Defaults to ['cat', 'dog']
                    clip_model:     Name of the CLIP model (see HuggingFace's model hub for available models)
                    clip_processor: Name of the CLIP processor (see HuggingFace's model hub for available models)
                    analysis_scope: Defines the scope of this annotator. This might be useful if you want to use
                                    this to (a) classify the whole scene or (b) classify previously detected object
                                    hypotheses.Defaults to CASViews.COLOR_IMAGE, which means that the annotator
                                    will classify the entire image.
                    filter_fn:      Function that can be used to filter the object hypotheses. Defaults to None.

                                        Example:
                                            ```python
                                                def filter_fn(object_hypothesis):
                                                    return object_hypothesis.classname == "cereal_box"
                                            ```

                                            This example would only classify object hypotheses that are
                                            cereal boxes. This might be useful if you want to classify
                                            only a subset of the object hypotheses.

                    save_top_k:     Number of top classes that will be annotated / saved in the CAS. Defaults to 5.
                """
                self.classes = ["Cat", "Dog"]
                # self.name = "ZeroShotClfAnnotator"
                self.clip_model = "openai/clip-vit-large-patch14"
                self.clip_processor = "openai/clip-vit-large-patch14"
                self.analysis_scope = CASViews.COLOR_IMAGE
                self.filter_fn: typing.Optional[
                    typing.Callable[[robokudo.types.scene.ObjectHypothesis], bool]
                ] = None
                self.save_top_k = min(5, len(self.classes))

        parameters = Parameters()

    def __init__(self, name="ZeroShotClfAnnotator", descriptor=Descriptor(), ) -> None:
        """Zero Shot Image Classification Model based on OpenAI's CLIP model."""
        self.parameters = descriptor.parameters
        super(ZeroShotClfAnnotator, self).__init__(name, descriptor)
        self.rk_logger.debug("Starting to init ZeroShotClfAnnotator")

        self.classes = self.parameters.classes
        self.model = CLIPModel.from_pretrained(self.parameters.clip_model)
        self.processor = CLIPProcessor.from_pretrained(self.parameters.clip_processor)
        self.analysis_scope = self.parameters.analysis_scope

        self.id2name = {str(i): name for i, name in enumerate(self.parameters.classes)}
        self.id2rgb = {
            i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for i, _ in enumerate(self.parameters.classes)
        }
        self.rk_logger.debug("Initialized ZeroShotClfAnnotator")

    @catch_and_raise_to_blackboard
    def compute(self):
        """Infer the classes for the given image and analysis scope.

        Raises:
            ValueError: If the analysis scope is not supported.
        """
        # GPSR Adjustments
        human_behaviours = ['standing', 'pointing', 'sitting', 'raising arm']
        colors = ['black', 'white', 'green']
        if self.parameters.gpsr:
            attributes = self.get_cas().get(CASViews.QUERY).obj.attribute
            if len(attributes) == 1:
                self.classes = human_behaviours
            else:
                self.classes = []
                if not (attributes[0] in colors):
                    colors.append(attributes)
                for color in colors:
                    self.classes.append(f'Person wearing {color} {attributes[1]}')
        # GPSR adjustments end
        start_timer = default_timer()

        if self.analysis_scope == CASViews.COLOR_IMAGE:
            self._analyze_scene()
        elif self.analysis_scope == robokudo.types.scene.ObjectHypothesis:
            self._analyze_object_hypotheses()
        else:
            raise ValueError(
                f"Analysis scope {self.analysis_scope} is not supported by the ZeroShotClfAnnotator."
                f"Choose between {CASViews.COLOR_IMAGE} and {robokudo.types.scene.ObjectHypothesis}."
            )

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.Status.SUCCESS

    def _analyze_scene(self):
        """Analyze the entire scene (predict classes for entire image)."""

        self.rk_logger.info("CLIP Annotator starts to analyze scene...")
        img = self.get_cas().get(CASViews.COLOR_IMAGE)
        img = Image.fromarray(img)

        inputs = self.processor(text=self.classes, images=img, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)

        results = []
        for label, prob in zip(self.classes, probs[0]):
            results.append((label, prob.item()))

        self.rk_logger.info("Image Classification Results:")
        self.rk_logger.info("Label \t\t Score")
        for top_k_idx, (label, prob) in enumerate(results):
            self.rk_logger.info(
                f"Top {top_k_idx} prediction: Class {label} with confidence\t{prob:.4f}"
            )

        # sort results by confidence
        results.sort(key=lambda x: x[1], reverse=True)

        # create classification annotation for top k results
        for top_k_idx in range(self.parameters.save_top_k):
            rk_classification = robokudo.types.annotation.Classification()
            rk_classification.classname = results[top_k_idx][0]
            rk_classification.confidence = results[top_k_idx][1]

            self.get_cas().annotations.append(rk_classification)

        # visualize the results
        visualization_img = self.get_cas().get(CASViews.COLOR_IMAGE)
        visualization_img = Image.fromarray(visualization_img)
        draw = PIL.ImageDraw.Draw(visualization_img)
        for top_k_idx in range(self.parameters.save_top_k):
            label = results[top_k_idx][0]
            prob = results[top_k_idx][1]
            text = f"{prob:.2f} /{label}"
            draw.text((10, 10 + 20 * top_k_idx), text, fill=(0, 0, 255))
        visualization_img = np.array(visualization_img)
        self.get_annotator_output_struct().set_image(visualization_img)

    def _cutout_oh_rois(self, object_hypotheses_list):
        """Cutout the object hypotheses' RoIs from the image"""

        img = self.get_cas().get(CASViews.COLOR_IMAGE)
        img = Image.fromarray(img)

        # Cutout the object hypotheses' RoIs from the image and predict the classes
        oh_cutouts = []
        for object_hypothesis in object_hypotheses_list:
            # self.rk_logger.critical(f"{type(object_hypothesis) = }")
            assert isinstance(object_hypothesis, robokudo.types.scene.ObjectHypothesis)
            assert object_hypothesis.roi is not None

            oh_cutout = img.crop(
                (
                    object_hypothesis.roi.roi.pos.x,
                    object_hypothesis.roi.roi.pos.y,
                    object_hypothesis.roi.roi.pos.x + object_hypothesis.roi.roi.width,
                    object_hypothesis.roi.roi.pos.y + object_hypothesis.roi.roi.height,
                )
            )

            oh_cutouts.append(oh_cutout)

        return oh_cutouts

    def _analyze_object_hypotheses(self):
        """Helper method to analyze the object hypotheses in the scene"""

        self.rk_logger.debug("CLIP Annotator starts to analyze object hypotheses...")
        object_hypotheses_list = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.ObjectHypothesis
        )

        if len(object_hypotheses_list) == 0:
            self.rk_logger.critical("Did not find any object hypotheses... ")
            return

        if self.parameters.filter_fn is not None:
            object_hypotheses_list = [
                oh for oh in object_hypotheses_list if self.parameters.filter_fn(oh)
            ]

        oh_cutouts = self._cutout_oh_rois(object_hypotheses_list)
        # Run CLIP on the batch of cutouts
        inputs = self.processor(
            text=self.classes, images=oh_cutouts, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).detach().numpy()

        # add the top prediction for each object hypothesis to the CAS
        for object_hypothesis, softmax_distribution in zip(object_hypotheses_list, probs):
            rk_classification = robokudo.types.annotation.Classification()
            rk_classification.classname = self.classes[softmax_distribution.argmax()]
            rk_classification.confidence = softmax_distribution.max().item()
            rk_classification.source = type(self).__name__
            object_hypothesis.classification = rk_classification  # TODO deprecated - Will be removed in future releases
            object_hypothesis.annotations.append(rk_classification)

            self.rk_logger.info(
                f"Classification results for Object hypothesis ID: '{object_hypothesis.id}'"
            )

            # save top k most confident predictions
            for pred_idx in range(self.parameters.save_top_k):
                rk_classification = robokudo.types.annotation.Classification()
                rk_classification.classname = self.classes[
                    softmax_distribution.argsort()[::-1][pred_idx]
                ]
                rk_classification.confidence = softmax_distribution[
                    softmax_distribution.argsort()[::-1][pred_idx]
                ].item()
                rk_classification.source = type(self).__name__
                object_hypothesis.annotations.append(rk_classification)

                self.rk_logger.info(
                    f"Top {pred_idx}: Classified as {rk_classification.classname} "
                    f"with confidence {round(rk_classification.confidence, 4)}"
                )
        self._visualize_objects(object_hypotheses_list)

    def _visualize_objects(self, object_hypotheses_list):
        """Visualize the object hypotheses in the image."""

        visualization_img = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        visualization_img = Image.fromarray(visualization_img)
        draw = PIL.ImageDraw.Draw(visualization_img)

        # TODO: change from oh_idx to the actual ID of the object hypothesis
        for oh_idx, object_hypothesis in enumerate(object_hypotheses_list):
            draw.rectangle(
                (
                    object_hypothesis.roi.roi.pos.x,
                    object_hypothesis.roi.roi.pos.y,
                    object_hypothesis.roi.roi.pos.x + object_hypothesis.roi.roi.width,
                    object_hypothesis.roi.roi.pos.y + object_hypothesis.roi.roi.height,
                ),
                outline=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                width=2,
            )

            label = object_hypothesis.classification.classname
            prob = object_hypothesis.classification.confidence
            text = f"{prob:.3f} /{label}"
            draw.text(
                (object_hypothesis.roi.roi.pos.x, object_hypothesis.roi.roi.pos.y - 20),
                text,
                fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            )

        visualization_img = np.array(visualization_img)
        self.get_annotator_output_struct().set_image(visualization_img)
