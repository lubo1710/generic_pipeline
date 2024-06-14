# Installation for GPSR
Overview:
+ Creating Workspace
+ Create python venv
+ Clone and setup Robokudo
+ Clone and setup generic_pipeline
+ First start

## Create Workspace for GPSR challenge

The first step is to create a workspace. Just follow the steps below.

```bash
source /opt/ros/noetic/setup.bash
mkdir -p ~/gpsr_ws/src
cd ~/gpsr_ws
catkin build
source ~/gpsr_ws/devel/setup.bash
```

That's it!

## Create python venv

This step creates a venv to protect this package for version conflicts with other installed packages.

```
cd ~/gpsr_ws/
python3 -m venv gpsr_venv
source gpsr_venv/bin/activate
```
Now you should be in this venv.

## Clone Robokudo

The next step is to clone Robokudo. To clone this repo via ssh you must have access to the repo.

```bash
cd ~/gpsr_ws/src
git clone --recurse-submodules git@gitlab.informatik.uni-bremen.de:robokudo/robokudo.git #ssh
cd ~/gpsr_ws/src/robokudo
rosdep install --from-path . --ignore-src 
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Clone generic_pipeline

This step clones the generic pipeline. In contrast to former version this package include all necessary
annotators. That implies that no further packages must be cloned in the installation process.

```bash
cd ~/gpsr_ws/src
git clone git@github.com:lubo1710/generic_pipeline.git
cd generic_pipeline
git checkout GPSR # Change Branch to GPSR
pip install -r requirements.txt
```

Now the src directory should contain only two packages called robokudo and generic_pipeline.

```bash
cd ~/gpsr_ws
catkin build
source devel/setup.bash
```

## First start

Start in the first terminal a roscore:

```bash
roscore
```

In a second terminal the generic_pipeline package:
```bash
source ~/gpsr_ws/gpsr_venv/bin/activate
source ~/gpsr_ws/devel/setup.bash
rosrun robokudo main.py _ae=generic _ros_pkg=generic_pipeline
```

After a few seconds two windows should pop up without content.

In a third terminal start axclient.py or use a high-level program for sending a query to robokudo.
A tutorial for that can be found under this link [link](http://wiki.ros.org/actionlib_tutorials/Tutorials/Calling%20Action%20Server%20without%20Action%20Client#Use_axclient_from_actionlib)

```bash
rosrun actionlib_tools axclient.py /robokudo/query
```

In the last terminal start a bagfile with objects. A very long with three objects can be found for
test purposes under this [link](https://nc.uni-bremen.de/index.php/s/9EKy8WCHnBAB9a6):

```bash
rosbag play /path/to/bagfile
```

Now open the window from axclient.py and send a query to robokudo. The result should be a list with 
with all detected objects and a pose

It is normal that in the first run a model will be downloaded. Consider at the robocup that this model is already
placed in the working directory before the challenges begins.