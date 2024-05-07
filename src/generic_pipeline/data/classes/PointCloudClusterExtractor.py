from dataclasses import dataclass
import robokudo.types.scene
@dataclass()
class Annotator:
    name = 'PointCloudClusterExtractor'
    source =  'robokudo.annotators.pointcloud_cluster_extractor'
    description = 'Detects objects based on a plane annotation an points above this plane'
    descriptor = {}
    parameters = {}
    inputs = [robokudo.cas.CASViews.CLOUD,robokudo.types.annotation.Plane]
    outputs = [robokudo.types.scene.ObjectHypothesis, robokudo.types.annotation.Classification]
    capabilities = {robokudo.types.annotation.Classification : 'object'}