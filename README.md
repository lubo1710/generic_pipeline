# Overview
This package contains a generic pipeline generation for Robokudo based on a directed graph. The work results from the bachelor thesis from Lukas Bollhorst in 2024 tutored by Patrick Mania.  
In general the implementation takes the incoming query and selects algorithms that provide annotations for this type and fulfills capabilities for the requested object. For example an object detection
algorithm based on machine learning will be only part of a pipeline if it is trained for the requested object.  
It is possible to expand or adapt the annoators that could be part of the pipeline by appending a dataclass file to the data directory. On branch main, there exists an example set of annotators but in general
i recommend to adapt the stack on the use-case.

# Installation
Just clone this package into your robokudo workspace under the *\src*. To learn how to create such a workspace, i recommend the [tutorial](https://robokudo.ai.uni-bremen.de/) of my tutor Patrick.
```bash
git clone git@github.com:lubo1710/generic_pipeline.git
catkin build
```

For GPSR change branch to GPSR.
It is important to rebuild your workspace after cloning the package to enable usability.
For following dependencies, i recommend to use a virtual environment.

```bash
pip install networkx
```
## How to add parameters to annotators
This package contains in the *data/classes* directory for every annotator a file with a dataclass. In this file you can adapt the parameters and descriptors for your own annotators. Just follow the pattern
from the currently existing files.
```python
from dataclasses import dataclass
import robokudo.types.scene
@dataclass()
class Annotator:
    name = 'ClusterColorAnnotator' # Class name of the annotator
    source =  'robokudo.annotators.cluster_color' # Describes the import specifications
    description = 'Detects color of ObjectHypothesis'
    descriptor = {} # Fill this dictionary where the key is a name of the variable and the value is the vlaue of the variable
    parameters = {} # Same approach 
    inputs = [robokudo.types.scene.ObjectHypothesis] # List with all required types
    outputs = [robokudo.types.annotation.SemanticColor] # List with all types that this annotator stores in the CAS
    capabilities = {robokudo.types.annotation.SemanticColor: ['red','yellow','green','blue','magenta', 
                                                                'cyan', 'white','black','grey']} # The capabilities of this annotator
```

In general your installation is ready, the next steps in this chapter include installations from further packages that are used in the presented stack of annotators. If you want to use your own stack of annotators, these
installations are not necessary.

## Install Yolov8 for Robokudo
The first step includes cloning the repo and installing the dependencies.
```bash
git clone git@gitlab.informatik.uni-bremen.de:robokudo/robokudo_yolo.git
cd robokudo_yolo
pip install -r requirements.txt
```
Through cloning the generic_pipeline package a weight and translation file already exists in your package.  

## Install robokudo_faces
The first step includes cloning the repo and installing the dependencies.
```bash
git clone git@gitlab.informatik.uni-bremen.de:robokudo/robokudo_faces.git
cd robokudo_faces
pip install -r requirements.txt
```

For gpsr are it is necessary to change the face_classification for new a new file.
It can be found under this [link](https://nc.uni-bremen.de/index.php/s/ZiqfWoWE9mooq7H).
Within the generic_pipeline package exists a directory called *faces* containing images from faces that can be recognized.
For GPSR the directory can be empty.

## Install ZeroShotClf
The first step includes cloning the repo and installing the dependencies.
```bash
git clone git@gitlab.informatik.uni-bremen.de:robokudo/robokudo_zero_shot_classification.git
cd robokudo_zero_shot_classification
pip install -r requirements.txt
catkin build
```

For GPSR:

Open the ZeroshotClfAnnotator and append at the top of the compute function the following code snippet

```python
    def compute(self):
        """Infer the classes for the given image and analysis scope.

        Raises:
            ValueError: If the analysis scope is not supported.
        """
        # GPSR Adjustments
        human_behaviours = ['standing', 'pointing', 'sitting', 'raising arm']
        colors = ['black', 'white', 'green']
        if self.parameters.gpsr:
            attributes = self.get_cas().get(CASViews.QUERY).obj.attribute
            if len(attributes) == 1:
                self.classes = human_behaviours
            else:
                self.classes = []
                if not (attributes[0] in colors):
                    colors.append(attributes)
                for color in colors:
                    self.classes.append(f'Person wearing {color} {attributes[1]}')
        # GPSR adjustments end

        ...
```

If you have trouble after this step try to execute the following command

```batch
pip install empy=3.3.4
```

## Install HumanDetection
The first step includes cloning the repo and installing the dependencies.
```bash
git clone git@gitlab.informatik.uni-bremen.de:robokudo/robokudo_human_detection.git
cd robokudo_human_detection
pip install -r requirements.txt
```

This package utilizes git large file system. make sure to install it before clonin gthis package.

The next step is stupid, but important for gpsr. Open the human_and_pose annotator and append at the end of the 
init function the following line

```python
def __init__(...):
    ...
    self.setup(10)
```

# How to start this package
Just run the following command:
```bash
rosrun robokudo main.py _ae=generic _ros_pkg=generic_pipeline
```
If your installation is correct, this engine waits for an incoming query. For test purposes use [actionlib_tools](git@github.com:ros/actionlib.git) or a high-level context.