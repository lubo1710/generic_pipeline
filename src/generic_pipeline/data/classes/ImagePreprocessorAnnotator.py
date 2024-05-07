from dataclasses import dataclass
import robokudo.cas


@dataclass()
class Annotator:
    name = 'ImagePreprocessorAnnotator'
    source =  'robokudo.annotators.image_preprocessor'
    description = 'Creates a Pointcloud out of color and depth image in the cas'
    descriptor = {}
    parameters = {}
    inputs = []
    outputs = [robokudo.cas.CASViews.CLOUD]
    capabilities = []