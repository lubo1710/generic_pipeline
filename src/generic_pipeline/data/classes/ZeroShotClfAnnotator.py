from dataclasses import dataclass
import robokudo.types.core
import robokudo.types.scene


@dataclass()
class Annotator:
    name = 'ZeroShotClfAnnotator'
    source =  'robokudo_zero_shot_classification.annotators.zero_shot_classification'
    description = 'Returns the more likely class out of the classes parameter'
    descriptor = {}
    parameters = {
        'classes' : ['dog', 'cat'],
        'save_top_k' : 0,
        'analysis_scope' : robokudo.types.scene.ObjectHypothesis
    }
    inputs = [robokudo.types.scene.ObjectHypothesis]
    outputs = [robokudo.types.core.Annotation]
    capabilities = {robokudo.types.core.Annotation : ['dog' , 'cat']}
