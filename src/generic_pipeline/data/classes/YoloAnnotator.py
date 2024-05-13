from dataclasses import dataclass
import robokudo


@dataclass()
class Annotator:
    name = 'YoloAnnotator'
    source =  'robokudo_yolo.annotators.YoloAnnotator'
    description = 'Detects objects in images and creates a ObjectHypothesis'
    descriptor = {}
    parameters = {
    'precision_mode' : True,
    'ros_pkg_path' : 'generic_pipeline',
    'weights_path' : 'src/generic_pipeline/Weights-SUTURO23/data_03_24.pt',
    'id2name_json_path' : 'src/generic_pipeline/data/json/id2name_edit.json',
    'threshold' : 0.6
    }
    inputs = []
    outputs = [robokudo.types.scene.ObjectHypothesis , robokudo.types.annotation.Classification]
    capabilities = {robokudo.types.annotation.Classification : ['Cup' , 'Metalmug', 'Crackerbox'],
                    robokudo.types.scene.ObjectHypothesis : ['Cup', 'Metalmug', 'Crackerbox']}