from dataclasses import dataclass
import robokudo.types.human
@dataclass()
class Annotator:
    name = 'StoreFaces'
    source =  'robokudo_faces.annotators.store_unknown_faces'
    description = 'Stores unknown faces'
    descriptor = {}
    parameters = {
        'ros_pkg_path' : 'generic_pipeline',
        'data_path' : 'src/generic_pipeline/data/faces',
        'noise' : 10,
    }
    inputs = [robokudo.types.annotation.Cuboid]
    outputs = [robokudo.types.annotation.Classification]
    capabilities = {robokudo.types.annotation.Classification : ['lukas', 'johannes']}