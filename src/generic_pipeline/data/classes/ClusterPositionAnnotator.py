from dataclasses import dataclass
import robokudo

@dataclass()
class Annotator:
    name = 'ClusterPositionAnnotator'
    source =  'robokudo.annotators.cluster_position'
    description = 'Calculates a pose based on a ObjectHypothesis'
    descriptor = {}
    parameters = {}
    inputs = [robokudo.types.scene.ObjectHypothesis]
    outputs = [robokudo.types.annotation.PositionAnnotation]
    capabilities = []