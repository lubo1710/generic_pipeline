from dataclasses import dataclass
import robokudo.types
import robokudo.types.scene
@dataclass()
class Annotator:
    name = 'FilterObjectsByRegion'
    source =  'generic_pipeline.annotators.filter_objects_by_region'
    description = 'Appends for each ObjectHypothesis a Location Annotation'
    descriptor = {}
    parameters = {
        'semantic_map_ros_package': 'robokudo',
        'semantic_map_name': 'suturo_project_room_long_table'
    }
    inputs = [robokudo.types.scene.ObjectHypothesis, robokudo.types.annotation.PoseAnnotation]
    outputs = [robokudo.types.annotation.LocationAnnotation]
    capabilities = {}