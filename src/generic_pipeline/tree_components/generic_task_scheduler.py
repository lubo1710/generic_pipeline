from typing import Optional
import py_trees
import robokudo.tree_components.task_scheduler
import robokudo.utils.tree
from robokudo.cas import CASViews
from generic_pipeline.utils.Network import RobokudoGraph


class GenericTaskScheduler(robokudo.tree_components.task_scheduler.TaskSchedulerBase,
                          robokudo.annotators.core.BaseAnnotator):
    """
    A Task Scheduler that submitted the query to the RobokudoGraph and receives a suitable sequence of annotators
    """

    def __init__(self, name="QueryBasedScheduler"):
        """
        Constructor that initializes the RobokudoGraph
        """
        self.graph = RobokudoGraph()
        super().__init__(name)


    def setup(self, timeout):
        return True

    def plan_new_job(self) -> Optional[py_trees.Sequence]:
        parent = self.parent
        assert (isinstance(parent, py_trees.composites.Sequence))

        self.graph.set_query(self.get_cas().get(CASViews.QUERY))
        new_job = self.graph.get_tree()

        return new_job
