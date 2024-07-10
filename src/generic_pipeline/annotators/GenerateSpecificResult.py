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
            right_color = False
            for oh_annotation in annotation.annotations:
                # Color
                if isinstance(oh_annotation, robokudo.types.annotation.SemanticColor):
                    if query_obj.color and not oh_annotation.color in query_obj.color and not right_color:
                        print(oh_annotation.color)
                        print(query_obj.color)
                        print('Not the right object due color')
                        queried = False
                        break
                    else:
                        right_color = True
                    object_designator.color.append(oh_annotation.color)

                # Classfications
                if isinstance(oh_annotation, robokudo.types.annotation.Classification):
                    # ZeroShot
                    if oh_annotation.source == 'ZeroShotClfAnnotator':
                        str = oh_annotation.classname.split(' ')
                        if len(str) > 1:
                            decision = str[2]
                        else:
                            decision = str[0]
                        if decision != query_obj.attribute[0]:
                            print('Not the right object due attributes')
                            queried = False
                            break
                        object_designator.attribute.append(oh_annotation.classname)
                        continue
                    # Face Classification
                    if oh_annotation.source == 'FaceClassification' or oh_annotation.source == 'StoreFaces':
                        object_designator.type = oh_annotation.classname
                        continue

                    # YoloAnnotator
                    print('Yolo beim auswerten')
                    if not self.is_required(oh_annotation):
                        print('Not the right object due classification')
                        queried = False
                        break
                    object_designator.type = oh_annotation.classname

                # Size
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
                    ps.header.frame_id = 'map'
                    object_designator.pose.append(ps)

                    object_designator.pose_source.append(oh_annotation.source)

                if isinstance(oh_annotation, robokudo.types.annotation.LocationAnnotation):
                    if oh_annotation.name != query_obj.location:
                        print('Not the right object due location')
                        queried = False
                        break
                    object_designator.location = oh_annotation.name

            if query_obj.location != '' and object_designator.location == '':
                queried = False
                print('Not the right object due location')


            if queried:
                query_result.append(object_designator)

        result.res = query_result
        blackboard.set(BBIdentifier.QUERY_ANSWER, result)

        self.feedback_message = f"Send result for {object_hypotheses_count} object hypotheses"
        return py_trees.Status.SUCCESS

    def is_required(self, annotation_type):
        print(annotation_type)

        tree_of_objects = {
                'cup' : ['Cupblue','Cupgreen','Cupsmall','Metalmug'],
                'muesli' : ['Crackerbox','Cerealbox','Mueslibox'],
                'fruit' : ['Strawberry','Apple','Orange','Pear','Lemon','Banana','Peach','Plum','Grapes',],
                'dish' : ['Metalplate', 'Metalbowl','Wineglass'],
                'cutlery' : ['Fork','Spoon','Knife'],
                'tool' : ['Scissors','Screwdriver','Clamp','Hammer','Woodenblock','Largemarker','Abrasivesponge'],
                'toy' : ['Rubikscube'],
                'ball' : ['Minisoccerball','Baseball','Softball','Tennisball'],
                'food' : ['Mustardbottle','Jellochocolatepuddingbox','Pringleschipscan','Jellobox','Sugarbox',
                          'Tomatosoupcan', 'Tunafishcan','Gelatinebox','Meatcan'],
                'drink' : ['Milk','Pitcher'],
                'coffee' : ['Coffeepack','Coffeecan','Masterchefcan'],
                'cleaning_tool' : ['Bleachcleanserbottle', 'Glasscleanerspraybottle','Dishwashertab','Scrubcleaner']
        }

        # Input from High level
        query_type = self.get_cas().get(CASViews.QUERY).obj.type

        # Annotation type
        annotation_type = annotation_type.classname

        # If empty or direct match object is required
        if query_type == '' or query_type == annotation_type:
            return True

        # Class instead of direct type
        if query_type in tree_of_objects.keys():
            if annotation_type in tree_of_objects[query_type]:
                return True

        # Not required because
        return False