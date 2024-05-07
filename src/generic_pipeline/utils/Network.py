import networkx as nx
import py_trees
import importlib.util
from itertools import chain, combinations
from networkx.classes import DiGraph
from networkx.algorithms.traversal import bfs_tree
import os
import robokudo


class RobokudoGraph(DiGraph):
    """
    After initialization the graph is empty. The tree will be set after the query is received. The tree only consist
    necessary edges and nodes. Edges represents annotators and nodes are inputs and outputs of annotators like
    SemanticColor or ObjectHypothesis. The annotator is stored under annotator and the name of the annotator is stored
    und name for example:
    self.add_general_edge('Mask','Pose', name='ClusterPose', annotator=ClusterPoseBBAnnotator())
    """
    def __init__(self, **attr):
        super().__init__(**attr)
        self.query = None
        self.specification = {}
        self.end_nodes = []


    def load_class_from_file(self,file_path, class_name):
        spec = importlib.util.spec_from_file_location(class_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name, None)

    def __set_tree(self, attributes):
        """
        Function will be called when the query is set. Must be called before get_tree
        Only contains edges that will be used during the generation progress.
        Name within the yaml file and filename must be equal.
        """
        try:
            # Laden der Yaml Dateien in Dictionary with name : capabilities
            yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/classes/')
            dicts = {}
            for filename in os.listdir(yaml_path):
                if not os.path.isfile(yaml_path + filename):
                    continue
                loaded_class = self.load_class_from_file(yaml_path + filename,'Annotator')
                instance = loaded_class()

                dependencies = {
                    'outputs' : instance.outputs,
                    'inputs' : instance.inputs,
                    'capabilities' : instance.capabilities
                }
                dicts[instance.name] = dependencies
        except FileNotFoundError:
            raise 'File not Found'

        failure = False
        while attributes and not failure: # Solange Attribute nicht leer
            failure = True
            for name in dicts.keys(): # laufe alle annotatoren durch
                for outputs in dicts[name]['outputs']: # laufe alle outputs von annotatoren durch
                    if outputs in attributes: # Falls gesuchter output drin ist
                        if outputs in self.specification.keys():  # Falls output genauer spezifiziert ist
                            element = self.specification[outputs]
                            if not element in dicts[name]['capabilities'][outputs]: # Falls gesuchtes Element nicht in den Capabilities des Annotators liegt überspringe
                                continue
                        # Not empty input
                        for inputs in dicts[name]['inputs']:
                            attributes.append(inputs)
                            for out in dicts[name]['outputs']:
                                # Füge alle inputs zu attributes hinzu und ziehe eine Kante
                                #attributes.append(inputs)
                                if out in self.specification.keys():
                                    element = self.specification[out]
                                    if not element in dicts[name]['capabilities'][
                                        out]:  # Falls gesuchtes Element nicht in den Capabilities des Annotators liegt überspringe
                                        continue
                                self.add_edge(out, inputs, name=name, annotator=self.__load_annotator(name + '.py'))
                                print(f'Add {name} to graph')
                        # Empty input
                        if not dicts[name]['inputs']:
                            for out in dicts[name]['outputs']:
                                if out in self.specification.keys():
                                    element = self.specification[out]
                                    if not element in dicts[name]['capabilities'][
                                        out]:  # Falls gesuchtes Element nicht in den Capabilities des Annotators liegt überspringe
                                        continue
                                self.add_edge(out,'root', name=name,annotator=self.__load_annotator(name + '.py'))
                                print(f'Add {name} to graph')
                        # Delete ouputs from the added annotator
                        for out in dicts[name]['outputs']:
                            while out in attributes:
                                print(out)
                                attributes.remove(out)
                            failure = False
        print(attributes)
        self.__reduce_graph()
        if failure:
            raise TypeError(f'Could not compute graph through {attributes}')


    def __load_annotator(self, filename):
        """
        Returns an annotator based on a path to the yaml file, which contains at least the name and source of the
        annotator
        """
        try:
            # Path to yaml file
            yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/classes/') + filename
            # Read from yaml
            loaded_class = self.load_class_from_file(yaml_path, 'Annotator')
            annotator = loaded_class()

            class_name = annotator.name  # class name
            class_path = annotator.source  # class path
            module = __import__(class_path, fromlist=[class_name])
            klass = getattr(module, class_name)
            # Dynamic class generation
            if not bool(annotator.parameters) and not bool(annotator.descriptor):
                instance = klass()
            else:
                descriptor = klass.Descriptor()
                for key, value in annotator.descriptor.items():
                    setattr(descriptor,key, value)
                for key, value in annotator.parameters.items():
                    setattr(descriptor.parameters, key, value)
                instance = klass(descriptor=descriptor)
            return instance

            # Catch FileNotFoundError
        except FileNotFoundError:
            raise 'Could not find the dataclass file'

    def __reduce_graph(self):
        '''
        This method reduces the graph through an algorithm, which minimalize the amount of used annotators.
        The problem is np-hard and can be solved by brute forcing.
        Man stellt alle Teilmengen von Annotatoren auf und führt dann BFS durch und überprüft, ob alle Knoten
        erreicht werden können. Resultat ist ein Baum mit minimaler Anzahl an Annotatoren
        '''
        annotators = set() # Mengen in Python
        all_edges = []
        for edge in self.edges():
            all_edges.append(edge)
            data = self.get_edge_data(edge[0],edge[1])
            annotators.add(data['name'])

        def get_subsets(s):
            return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

        subsets = list(get_subsets(annotators))
        annotator_subset = []
        for subset in subsets:
            # compute subgraph
            graph = nx.DiGraph()
            graph.add_nodes_from(self.nodes())
            for edge in all_edges:
                if self.get_edge_data(edge[0], edge[1])['name'] in subset:
                    graph.add_edge(edge[1],edge[0])
            # BFS in subgraph
            tree = bfs_tree(graph, 'root')
            if len(tree.nodes()) == len(graph.nodes):
                annotator_subset = subset
                break
        # Delete edges, which are not needed
        list_of_edges = []
        for edge in self.edges():
            edge_name = self.get_edge_data(edge[0],edge[1])['name']
            if not edge_name in annotator_subset:
                list_of_edges.append(edge)

        for edge in list_of_edges:
            self.remove_edge(edge[0], edge[1])

        print(f'Reduced Graph by {len(list_of_edges)} edges')


    def get_tree(self):
        """
        Returns a Sequence of Annotators in semantic correct order based on the corresponding graph.
        """
        # Topologische Sortierung durchführen
        topological_order = list(nx.topological_sort(self))
        # Kanten entsprechend der topologischen Sortierung extrahieren
        sorted_edges = []
        for node in topological_order:
            successors = list(self.successors(node))
            sorted_edges.extend([(node, succ) for succ in successors])
        # Reverse list cause direction of edges is shown to input
        sorted_edges = list(reversed(sorted_edges))

        annotator_list = []
        annotator_names = []
        for edges in sorted_edges:
            annotator = self.get_edge_data(edges[0], edges[1])
            if annotator['name'] in annotator_names:
                continue
            annotator_names.append(annotator['name'])
            annotator_list.append(annotator['annotator'])

        print( f'Network compute this pipeline: {annotator_names}')
        seq = py_trees.composites.Sequence()
        seq.add_children(annotator_list)
        return seq

    def set_query(self, query):
        """
        Reads the query and stores the required attributes in a list, which will be used to be starting nodes from
        th RobokudoGraph.

        string type
        string description
        ObjectDesignator obj
            string uid
            string type -> Classification
            string[] shape
            ShapeSize[] shape_size -> Shape
            string[] color  -> SemanticColor
            string location ->  PositionAnnotation (Region Filter)
            string size
            geometry_msgs/PoseStamped[] pose   -> PoseAnnotation
            string[] pose_source

            string[] attribute  -> ZeroShot opportunities

            string[] description
        """

        self.query = query
        # Add pose because the query always starts with 'DETECT ...'
        queried_attributes = [robokudo.types.annotation.PoseAnnotation]
        if query.obj.color:
            self.specification[robokudo.types.annotation.SemanticColor] = query.obj.color[0].lower()
            queried_attributes.append(robokudo.types.annotation.SemanticColor)
        if query.obj.type != '':
            self.specification[robokudo.types.annotation.Classification] = query.obj.type.lower()
            queried_attributes.append(robokudo.types.annotation.Classification)
        else:
            self.specification[robokudo.types.annotation.Classification] = 'object'
            queried_attributes.append(robokudo.types.annotation.Classification)
        if query.obj.shape_size:
            queried_attributes.append(robokudo.types.annotation.Shape)
        if query.obj.location != '':
            queried_attributes.append(robokudo.types.annotation.LocationAnnotation)
        if query.obj.attribute:
            self.specification[robokudo.types.core.Annotation] = query.obj.attribute[0].lower()
            queried_attributes.append(robokudo.types.core.Annotation)
        self.end_nodes = queried_attributes
        self.__set_tree(queried_attributes)