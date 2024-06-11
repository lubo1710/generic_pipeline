from dataclasses import dataclass
import robokudo.types.human
@dataclass()
class Annotator:
    name = 'FaceDetector'
    source =  'robokudo_faces.annotators.face_detector'
    description = 'Detect faces'
    descriptor = {}
    parameters = {}
    inputs = []
    outputs = [robokudo.types.human.FaceAnnotation, robokudo.types.scene.HumanHypothesis]
    capabilities = {}