from dataclasses import dataclass
import robokudo.types.human
@dataclass()
class Annotator:
    name = 'FaceClassification'
    source =  'robokudo_faces.annotators.face_classification'
    description = 'Classifies faces'
    descriptor = {}
    parameters = {
        'ros_pkg_path' : 'milestone1',  # If set, use use data_path as a relative path to self.ros_pkg_path
        'data_path' : 'src/milestone1/faces',  # Relative Path to the folder containing the models
        'file_names' : ['0.png','1.png','2.png'],  # files in self.data_path to load
        'labels' : ['0','1','2']  # 'class labels' for each of the file
    }
    inputs = [robokudo.types.human.FaceAnnotation]
    outputs = [robokudo.types.annotation.Classification]
    capabilities = {robokudo.types.annotation.Classification : ['0','1','2']}