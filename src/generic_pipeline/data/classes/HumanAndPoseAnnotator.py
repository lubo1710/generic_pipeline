from dataclasses import dataclass
import robokudo.types.human
@dataclass()
class Annotator:
    name = 'HumanAndPoseAnnotator'
    source =  'robokudo_human_detection.annotators.human_and_pose'
    description = 'Detect humans'
    descriptor = {}
    parameters = {
        'labels' : [],
        'yolo_model_name' : 'yolov8n-pose.pt',
        'ros_pkg' : 'generic_pipeline',
        'model_path' : 'src/generic_pipeline/Weights-SUTURO23',
        'shrink_bounding_box' : None,
        'shrink_as_annotation' : False,
        'generate_clouds_for_keypoints' : False,
        'keypoint_confidence_threshold_for_clouds' : 0.6,
        'keypoint_masking_radius' : 6,
        'warmup_annotator' : True,
        'depth_truncate' : 3.5
        }
    inputs = []
    outputs = [robokudo.types.scene.ObjectHypothesis]
    capabilities = {robokudo.types.scene.ObjectHypothesis : ['person']}