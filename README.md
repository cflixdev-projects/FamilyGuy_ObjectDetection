# Yolo26 FamilyGuy Weights

> [!IMPORTANT]
> Given weights only inlcude for characters such as: `Peter, Stewie, Brian & Lois.` Both weights were traind with the `same dataset` for respective architectures such as yolo26 & cnn. 

## Weights
* `[best.pt]` yolo26 architecture
* `[cnn-weights/family_guy_best.pth]` cnn architecture

## Setup

## Prerequisites 
This projects requires Python 3.11.9 or higher. It is recommended to use a virtual environment to avoid dependency conflicts.
With the standard Python installation, you can create and activate a virtual environment using the following commands:

```bash
python -m venv .venv
source .venv/bin/activate
```

## Installation

```bash
git clone https://github.com/cflixdev-projects/yolov26-family-guy-weights.git
cd cflixdev-projects/yolov26-family-guy-weights
pip install -r requirements.txt
```

**Packages:**

```bash
torch
torchvision
pillow
opencv-python
numpy
mss
ultralytics
```

