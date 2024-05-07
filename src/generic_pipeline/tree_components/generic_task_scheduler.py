from typing import Optional
import py_trees
import robokudo.tree_components.task_scheduler
import robokudo.utils.tree
from robokudo.cas import CASViews
from generic_pipeline.utils.Network import RobokudoGraph


class GenericTaskScheduler(robokudo.tree_components.task_scheduler.TaskSchedulerBase,
                          robokudo.annotators.core.BaseAnnotator):
    """
    A Task Scheduler that checks the active Query in the CAS to infer which perception subtree to execute.
    You can apply a function to infer per use-case which perception tree you want to incorporate.

    Original implementation by Malte Huerkamp
    """

    def __init__(self, name="QueryBasedScheduler"):
        """
        Tasks should be a dict with key='task-identifier' and value a py_trees.Behaviour)

        :param: filter_fn a callable/function which returns a string with the identifier of the subtree to include.
        This function will receive the CASViews.QUERY and can then decide which subtree identifier is the desired one.
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
