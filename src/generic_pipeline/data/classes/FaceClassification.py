from dataclasses import dataclass
import robokudo.types.human
@dataclass()
class Annotator:
    name = 'FaceClassification'
    source =  'generic_pipeline.annotators.face_classification'
    description = 'Classifies faces'
    descriptor = {}
    parameters = {
        'ros_pkg_path' : 'generic_pipeline',  # If set, use data_path as a relative path to self.ros_pkg_path
        'data_path' : 'src/generic_pipeline/data/faces',  # Relative Path to the folder containing the models
        'file_names' : ['leonie.png', 'lukas.png'],  # files in self.data_path to load
        'labels' : ['leonie','lukas']  # 'class labels' for each of the file
    }
    inputs = [robokudo.types.human.FaceAnnotation]
    outputs = [robokudo.types.annotation.Cuboid]
    capabilities = {}