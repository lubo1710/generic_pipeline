from dataclasses import dataclass
import robokudo

@dataclass()
class Annotator:
    name = 'ClusterPositionAnnotator'
    source =  'robokudo.annotators.cluster_position'
    description = 'Calculates a pose based on a ObjectHypothesis'
    descriptor = {}
    parameters = {}
    inputs = [robokudo.types.scene.HumanHypothesis, robokudo.cas.CASViews.CLOUD]
    outputs = [robokudo.types.annotation.PoseAnnotation]
    capabilities = {robokudo.types.annotation.PoseAnnotation : ['person']}