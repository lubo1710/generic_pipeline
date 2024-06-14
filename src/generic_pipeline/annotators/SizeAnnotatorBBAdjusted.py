from dataclasses import dataclass
import robokudo.types.human
@dataclass()
class Annotator:
    name = 'SizeAnnotatorBBAdjusted'
    source =  'milestone1.annotators.size_annotator_adjusted'
    description = 'Estimates size'
    descriptor = {}
    parameters = {}
    inputs = [robokudo.types.scene.ObjectHypothesis]
    outputs = [robokudo.types.annotation.Shape]
    capabilities = {}