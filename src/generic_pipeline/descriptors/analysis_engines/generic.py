import robokudo.analysis_engine
import py_trees
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
import robokudo.descriptors.camera_configs.config_hsr
import robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform
import robokudo.io.camera_interface
import robokudo.idioms
from robokudo.annotators.query import QueryAnnotator
import robokudo.annotators
from generic_pipeline.annotators.GenerateSpecificResult import GenerateSpecificResult
from generic_pipeline.tree_components.generic_task_scheduler import GenericTaskScheduler

class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self):
        return "milestone1_demo"

    def implementation(self):
        """
        Creates a generic pipeline based on the entries within the received query.
        """
        hsr_camera_config = robokudo.descriptors.camera_configs.config_hsr.CameraConfig()
        hsr_config = CollectionReaderAnnotator.Descriptor(
            camera_config=hsr_camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(hsr_camera_config))

        wrapper = py_trees.composites.Sequence(name='Wrapper', memory=True)
        wrapper.add_child(GenericTaskScheduler())

        seq = robokudo.pipeline.Pipeline('RWPipeline')

        # Basic pipeline, where wrapper contains specific annotators for query
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                QueryAnnotator(),
                CollectionReaderAnnotator(hsr_config),
                wrapper,
                GenerateSpecificResult()
            ]
        )
        return seq