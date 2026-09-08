# Object Detection using deep learning models on Family Guy Characters
## used models: YOLO26 & CNN 

> [!IMPORTANT]
> Given weights only inlcude for characters such as: `Peter, Stewie, Brian & Lois.` Both weights were traind with the `same dataset` for respective architectures such as YOLO26 & cnn. 

## Weights
* `[best.pt]` YOLO26 architecture
* `[cnn-weights/family_guy_best.pth]` CNN architecture

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
pip install torch
pip install torchvision
pip install pillow
pip install opencv-python
pip install numpy
pip install mss
pip install ultralytics
```

```python
class FamilyGuyCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```


