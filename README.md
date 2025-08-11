# ComputerVisionProject
this repo is dedicated to Computer Vision's project with Prof. Nicola Conci

In the domain of video surveillance and human behavior analysis, human motion plays a major role. In particular the displacement of each subject with respect to his/her neighbours and the environment can be revealing in terms of the ongoing activities/behaviors. 

The aim of the project is to analyze human trajectories, taken from the Stanford Drone Dataset (small version available here on Kaggle), a dataset collected by a UAV surveilling different areas of the Stanford Campus. 

For the project you will consider the scenes “video0” and “video3” from the “little” set. 


# You are asked to provide the following information: 

- Implement a detection and tracking system for the moving objects;
- Compute the trajectories for the detected moving objects;
- Compute an accuracy metric of the obtained trajectories, using for example the Mean Squared Error;
- Considering all objects, based on the ground truth, what is the most frequent path? (e.g. objects enter left and exit top)

# Bonus Task: 

- At some point, the roundabout is not visible, thus for each object you have the entry point and the exit point. You are asked to reconstruct the missing trajectory based on past observations and/or prior knowledge. For this task, you can process the ground truth annotations. 


# Human Motion Analysis – Stanford Drone Dataset

## Overview
This project implements a detection + tracking system to analyze human trajectories from UAV footage (`video0` and `video3` from the "little" set of the Stanford Drone Dataset).

## Pipeline
1. **Detection** – YOLOv8 (anchor-free, small object friendly)
2. **Tracking** – ByteTrack for robust ID continuity
3. **Trajectory Computation** – Extract center coordinates per frame per object
4. **Evaluation** – Mean Squared Error vs. ground truth
5. **Most Frequent Path** – Entry/exit mapping
6. **Bonus** – Missing trajectory reconstruction via interpolation

## How to Run
```bash
pip install -r requirements.txt
python run_pipeline.py
