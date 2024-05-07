from dataclasses import dataclass
import robokudo
import robokudo.types
@dataclass()
class Annotator:
    name = 'PlaneAnnotator'
    source =  'robokudo.annotators.plane'
    description = 'Finds the biggest plane in a Cloud'
    descriptor = {}
    parameters = {
        'visualize_plane_model' : True
    }
    inputs = [robokudo.cas.CASViews.CLOUD]
    outputs = [robokudo.types.annotation.Plane]
    capabilities = {}