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
It can be found under this link TODO.
Within the generic_pipeline package exists a directory called *faces* containing images from faces that can be recognized.
For GPSR the directory can be empty.

## Install ZeroShotClf
The first step includes cloning the repo and installing the dependencies.
```bash
git clone git@gitlab.informatik.uni-bremen.de:robokudo/robokudo_zero_shot_classification.git
cd robokudo_zero_shot_classification
pip install -r requirements.txt
```

If you have trouble after this step try to execute the following command

```batch
pip install empy=3.3.4
```

# How to start this package
Just run the following command:
```bash
rosrun robokudo main.py _ae=generic _ros_pkg=generic_pipeline
```
If your installtion is correct, this engine waits for a incoming query. For test purposes use [actionlib_tools](git@github.com:ros/actionlib.git) or a high-level context.

