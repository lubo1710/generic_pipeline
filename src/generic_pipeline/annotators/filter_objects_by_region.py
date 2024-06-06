import py_trees
import robokudo.annotators.core
from robokudo.cas import CASViews
import robokudo.defs
from robokudo.types.annotation import PoseAnnotation
from robokudo.utils.annotator_helper import transform_pose_from_cam_to_world
from robokudo.utils.module_loader import ModuleLoader
import robokudo.types.scene
class FilterObjectsByRegion(robokudo.annotators.core.BaseAnnotator):
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.semantic_map_ros_package = "robokudo"
                self.semantic_map_name = "semantic_map_iai_kitchen"  # should be in descriptors/semantic_maps/

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="FilterObjectsByRegion", descriptor=Descriptor()):
        super().__init__(name=name, descriptor=descriptor)
        self.semantic_map = None
        self.load_semantic_map()
    def load_semantic_map(self) -> None:
        module_loader = ModuleLoader()
        self.semantic_map = module_loader.load_semantic_map(self.descriptor.parameters.semantic_map_ros_package,
                                                            self.descriptor.parameters.semantic_map_name)
    def update(self):
        query = None
        if self.get_cas().contains(CASViews.QUERY):
            query = self.get_cas().get(CASViews.QUERY)

        self.load_semantic_map()
        if query is None:
            return py_trees.Status.SUCCESS
        else:
            queried_location = query.obj.location
            if queried_location == "":
                return py_trees.Status.SUCCESS
            else:
                self.rk_logger.info(f"Setting filter to check for location '{queried_location}'")
                active_regions = dict()
                try:
                    active_regions[queried_location] = self.semantic_map.entries[queried_location]
                except KeyError as ke:
                    raise Exception(f"Couldn't find requested location {queried_location} in semantic map")

        region_position_and_size = {}
        region_name = ""
        for key, region in active_regions.items():
            assert (isinstance(region, robokudo.semantic_map.SemanticMapEntry))
            region_position_and_size = {"position": [region.position_x, region.position_y, region.position_z], "size": [region.x_size, region.y_size,region.z_size]}
            region_name = key
        region_annotation = robokudo.types.annotation.LocationAnnotation()
        region_annotation.name = region_name

        annotations = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)
        for annotation in annotations:
            for oh_annotation in annotation.annotations:
                if isinstance(oh_annotation, robokudo.types.annotation.PoseAnnotation) or isinstance(oh_annotation, robokudo.types.annotation.PositionAnnotation):
                    pose_map = transform_pose_from_cam_to_world(self.get_cas(), oh_annotation)
                    position = [pose_map.translation[0], pose_map.translation[1], pose_map.translation[2]]
                    if self.postion_in_region(position, region_position_and_size):
                        annotation.annotations.append(region_annotation)
                        break

        return py_trees.Status.SUCCESS

    def postion_in_region(self,position_of_object, region_position_and_size):
        i = 0
        while i <= 2:
            region_coordinate = region_position_and_size["position"][i]
            region_size = region_position_and_size["size"][i]
            if region_size < 0:
                region_range = [region_coordinate+region_size,region_coordinate]
            else:
                region_range = [region_coordinate, region_coordinate + region_size]
            if position_of_object[i] >= region_range[0] and position_of_object[i] <= region_range[1]:
                i += 1
                continue
            else:
                return False

            i += 1
        return True











