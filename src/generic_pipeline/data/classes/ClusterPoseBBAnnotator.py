from dataclasses import dataclass
import robokudo.types.scene

@dataclass()
class Annotator:
    name = 'ClusterPoseBBAnnotator'
    source =  'robokudo.annotators.cluster_pose_bb'
    description = 'Calculates a pose based on a ObjectHypothesis'
    descriptor = {}
    parameters = {}
    inputs = [robokudo.cas.CASViews.CLOUD, robokudo.types.scene.ObjectHypothesis]
    outputs = [robokudo.types.annotation.PoseAnnotation]
    capabilities = {robokudo.types.annotation.PoseAnnotation : ['object']}