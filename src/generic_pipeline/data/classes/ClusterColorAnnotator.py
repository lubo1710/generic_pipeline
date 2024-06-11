from dataclasses import dataclass
import robokudo.types.scene
@dataclass()
class Annotator:
    name = 'ClusterColorAnnotator'
    source =  'robokudo.annotators.cluster_color'
    description = 'Detects color of ObjectHypothesis'
    descriptor = {}
    parameters = {}
    inputs = [robokudo.types.scene.ObjectHypothesis]
    outputs = [robokudo.types.annotation.SemanticColor]
    capabilities = {}

