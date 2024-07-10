from typing import Optional
import py_trees
import robokudo.tree_components.task_scheduler
import robokudo.utils.tree
from robokudo.cas import CASViews
from generic_pipeline.utils.Network import RobokudoGraph
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from io import BytesIO
import os

os.environ["QT_API"] = "pyqt5"

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
        ''' Plans new job with the network. '''
        parent = self.parent
        assert (isinstance(parent, py_trees.composites.Sequence))

        self.graph.set_query(self.get_cas().get(CASViews.QUERY))
        new_job = self.graph.get_tree()
        self.vis()
        return new_job

    def vis(self):
        ''' Visualizes the graph. '''
        # Clear Plot
        plt.clf()

        # map names to increase size of node names
        mapping = {}
        for node in self.graph:
            name = self.clean_name(node)
            mapping[node] = name

        # Overwrite nodes
        self.graph = nx.relabel_nodes(self.graph, mapping)

        # figure out name and set layout
        edge_names = nx.get_edge_attributes(self.graph, 'name')
        pos = nx.circular_layout(self.graph)  # Position nodes using Fruchterman-Reingold force-directed algorithm

        # Actual drawing of the graph
        print(self.graph.edges)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', arrows=True)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_names)

        # Convert to an array for visualization
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert the BytesIO object to a numpy array
        img = Image.open(buf)
        img_array = np.array(img)
        self.get_annotator_output_struct().set_image(img_array)
        pass

    def clean_name(self,name):
        ''' Remove non-alphabetic chars from a submitted string. '''
        complete_as_string = str(name)
        last_component = complete_as_string.split('.')[-1]
        cleaned_string = ''.join(char for char in last_component if char.isalpha())
        print(cleaned_string)
        return cleaned_string
