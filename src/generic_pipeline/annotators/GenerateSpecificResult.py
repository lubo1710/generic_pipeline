import copy
import logging
import queue

import actionlib
import py_trees
import rospy
import robokudo_msgs.msg
import geometry_msgs
import robokudo.annotators.core
from robokudo.identifier import BBIdentifier
from robokudo.cas import CASViews
import robokudo.defs
from robokudo.types.annotation import PoseAnnotation, PositionAnnotation
from robokudo.utils.annotator_helper import transform_pose_from_cam_to_world
from robokudo_msgs.msg import ShapeSize
from geometry_msgs.msg import Vector3

class GenerateSpecificResult(robokudo.annotators.core.BaseAnnotator):
    """
    This class generates a specific result for the generic pipeline package.
    """

    def __init__(self, name="GenerateQueryResult"):
        super().__init__(name=name)

    def update(self):
        blackboard = py_trees.Blackboard()
        annotations = self.get_cas().annotations
        object_hypotheses_count = 0
        query_result = []

        query_obj = self.get_cas().get(CASViews.QUERY).obj

        result = robokudo_msgs.msg.QueryResult()
        for annotation in annotations:
            if not isinstance(annotation, robokudo.types.scene.ObjectHypothesis):
                continue
            object_hypotheses_count += 1
            object_designator = robokudo_msgs.msg.ObjectDesignator()
            queried = True
            for oh_annotation in annotation.annotations:
                print(type(oh_annotation))
                if isinstance(oh_annotation, robokudo.types.annotation.SemanticColor):
                    if query_obj.color and not oh_annotation.color in query_obj.color:
                        print('Not the right object due color')
                        print(f'{oh_annotation.color} is not in {query_obj.color}')
                        queried = False
                        break
                    object_designator.color.append(oh_annotation.color)

                if isinstance(oh_annotation, robokudo.types.annotation.Classification):
                    if oh_annotation.source == 'ZeroShotClfAnnotator':
                        if oh_annotation.classname != query_obj.attribute[0]:
                            print('Not the right object due attributes')
                            queried = False
                            break
                        object_designator.attribute.append(oh_annotation.classname)
                        continue
                    if oh_annotation.classname != query_obj.type and query_obj.type != '':
                        print('Not the right object due classification')
                        queried = False
                        break
                    print(oh_annotation.source)
                    object_designator.type = oh_annotation.classname

                if isinstance(oh_annotation, robokudo.types.cv.BoundingBox3D):
                    size = ShapeSize()
                    vector = Vector3()
                    vector.x = oh_annotation.x_length
                    vector.y = oh_annotation.y_length
                    vector.z = oh_annotation.z_length
                    size.dimensions = vector
                    size.radius = 0
                    object_designator.shape_size.append(size)

                if isinstance(oh_annotation, robokudo.types.annotation.PoseAnnotation):
                    ps = geometry_msgs.msg.PoseStamped()

                    pose_map = transform_pose_from_cam_to_world(self.get_cas(), oh_annotation)
                    # TODO create PoseStamped and add it to the list
                    ps.pose.position.x = pose_map.translation[0]
                    ps.pose.position.y = pose_map.translation[1]
                    ps.pose.position.z = pose_map.translation[2]

                    ps.pose.orientation.x = pose_map.rotation[0]
                    ps.pose.orientation.y = pose_map.rotation[1]
                    ps.pose.orientation.z = pose_map.rotation[2]
                    ps.pose.orientation.w = pose_map.rotation[3]

                    # We assume that the pose annotation is in CAMERA coordinates
                    ps.header = copy.deepcopy(self.get_cas().get(CASViews.CAM_INFO).header)
                    ps.header.frame_id = '/map'
                    object_designator.pose.append(ps)

                    object_designator.pose_source.append(oh_annotation.source)

                if isinstance(oh_annotation, robokudo.types.annotation.PositionAnnotation):
                    # TODO: Add here check with region filter
                    ps = geometry_msgs.msg.PoseStamped()

                    pos = PoseAnnotation()
                    pos.source = oh_annotation.source
                    pos.translation.insert(0, oh_annotation.translation[0])
                    pos.translation.insert(1, oh_annotation.translation[1])
                    pos.translation.insert(2, oh_annotation.translation[2])

                    pos.rotation.insert(0, 0)
                    pos.rotation.insert(1, 0)
                    pos.rotation.insert(2, 0)
                    pos.rotation.insert(3, 1)

                    pose_map = transform_pose_from_cam_to_world(self.get_cas(), pos)

                    # TODO create PoseStamped and add it to the list
                    ps.pose.position.x = pose_map.translation[0]
                    ps.pose.position.y = pose_map.translation[1]
                    ps.pose.position.z = pose_map.translation[2]

                    ps.pose.orientation.x = 0
                    ps.pose.orientation.y = 0
                    ps.pose.orientation.z = 0
                    ps.pose.orientation.w = 1

                    # We assume that the pose annotation is in CAMERA coordinates
                    ps.header = copy.deepcopy(self.get_cas().get(CASViews.CAM_INFO).header)
                    ps.header.frame_id = '/map'
                    print('Füge Pose hinzu')
                    object_designator.pose.append(ps)

                    object_designator.pose_source.append(oh_annotation.source)

            if queried:
                query_result.append(object_designator)

        result.res = query_result
        blackboard.set(BBIdentifier.QUERY_ANSWER, result)

        self.feedback_message = f"Send result for {object_hypotheses_count} object hypotheses"
        return py_trees.Status.SUCCESS

