import numpy as np
import py_trees
import robokudo.annotators.core
import open3d as o3d

from robokudo.cas import CASViews
import robokudo.utils.error_handling
from timeit import default_timer

"""
This module implements a statistical outlierremoval based on the standard deviation and
number of neighbors in a point cloud. Afterwards a clustering is applied, to get for each 
object-hypothesis the actual object. This implementation works in place, such that there is no 
remaining copy of the old state for some point cloud. 

Authors: Sorin Arion, Naser Azizi
"""


class OutlierRemovalOnObjectHypothesisAnnotator(robokudo.annotators.core.ThreadedAnnotator):
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.dbscan_neighbors = 90
                self.dbscan_epsilon = 0.02
                self.stat_neighbors = 200
                self.stat_std = 0.5

        parameters = Parameters()

    def __init__(self, name="OutlierRemovalOnObjectHypothesis", descriptor=Descriptor()):
        """
        Default construction. Minimal one-time init!
        """
        super(OutlierRemovalOnObjectHypothesisAnnotator, self).__init__(name, descriptor)
        #super(YoloAnnotator, self).__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def compute(self) -> py_trees.Status:
        start_timer = default_timer()
        pcd_cluster = self.cluster_statistical_outlierremoval_pcd()

        if not pcd_cluster:
            self.rk_logger.warning(f"No Clusters have been found.")
            self.feedback_message = f"No clusters have been found"
            raise Exception("No Clusters have been found.")
        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS

    def cluster_statistical_outlierremoval_pcd(self):
        """
        Perform outlier removal and clustering on each object hypothesis.

        :return: True, if atleast one of the object hypotheses could be optimized. False otherwise.
        """
        print('Fängt an')
        annotations = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)
        print(annotations)
        vis_geometries = []

        optimized_one_cluster = False

        for annotation in annotations:
            if annotation.points is None:
                continue

            pcd = annotation.points
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=self.descriptor.parameters.stat_neighbors,
                                                     std_ratio=self.descriptor.parameters.stat_std)

            pcd = pcd.select_by_index(ind)

            with o3d.utility.VerbosityContextManager(
                    o3d.utility.VerbosityLevel.Debug) as cm:
                labels = np.array(
                    pcd.cluster_dbscan(eps=self.descriptor.parameters.dbscan_epsilon,
                                       min_points=self.descriptor.parameters.dbscan_neighbors, print_progress=True))
            """
            We pick the biggest cluster, assuming that its point cloud represents
            an actual object and not noise
            """
            try:
                cluster_sizes = []
                max_label = labels.max()
                for val in range(0, max_label + 1):
                    cluster_sizes.append(np.where(labels == val)[0].shape[0])
                best_cluster = np.argmax(np.asarray(cluster_sizes))

            except ValueError as e:
                # We couldn't optimize THIS object hypothesis, but maybe there are other ones that already have been
                # optimized in previous iterations or we will find optimizables ones next
                continue
                # return False

            optimized_one_cluster = True

            cluster_indices = np.where(labels == best_cluster)[0]
            clustered_pcd = pcd.select_by_index(cluster_indices)
            # Replace old state with clustered point cloud
            annotation.points = clustered_pcd
            vis_geometries.append(clustered_pcd)

        if not optimized_one_cluster:
            return False

        visualization_img = self.get_cas().get(CASViews.COLOR_IMAGE)
        self.get_annotator_output_struct().set_image(visualization_img)
        self.get_annotator_output_struct().set_geometries(vis_geometries)

        return True
