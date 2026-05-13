# Driver Drowsiness Detection using Deep Learning

A real-time Driver Drowsiness Detection System built using Deep Learning and Computer Vision techniques.  
This project uses CNN-based models along with MediaPipe Face Mesh for eye detection and monitoring driver alertness in real time.

---

# Features

- Real-time drowsiness detection using webcam
- Eye state classification (Open / Closed)
- MediaPipe Face Mesh for facial landmark detection
- CNN-based deep learning models
- Alarm alert when drowsiness is detected
- Training and dataset preparation scripts included

---

# Project Structure

```bash
.
├── models/
├── create_dataset.py
├── enumerate_data.py
├── extract_frames.py
├── realtime.py
├── train_model.py
├── DL_Project_PPT.pdf
├── Project_Report.pdf
├── Training_Statistics.jpg
├── alarm.mp3
├── applsci-13-07849.pdf
├── dataset_github.rar
├── requirements.txt
```

---

# File Description

| File | Description |
|------|-------------|
| `create_dataset.py` | Creates dataset for training |
| `enumerate_data.py` | Organizes and enumerates dataset |
| `extract_frames.py` | Extracts frames from videos/images |
| `train_model.py` | Trains CNN models |
| `realtime.py` | Runs real-time drowsiness detection |
| `models/` | Contains trained model files |
| `alarm.mp3` | Alarm sound for drowsiness alert |
| `requirements.txt` | Required Python libraries |
| `Project_Report.pdf` | Complete project report |
| `DL_Project_PPT.pdf` | Project presentation |

---

# Models Used

The following CNN architectures were experimented with:

- InceptionV3
- VGG16
- ResNet50V2

Among these, **ResNet50V2** achieved the best accuracy and robustness.

---

# Dataset

The dataset used for training and evaluation can be downloaded from:

https://datasets.esdalab.ece.uop.gr/download-files/

After downloading, extract and organize the dataset before running the training scripts.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Chitranshu-DTU-Official/Deep-Learning-Project.git
cd Deep-Learning-Project
```

---

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the real-time drowsiness detection system using:

```bash
python realtime.py
```

The webcam will open and the system will monitor eye state in real time.

If drowsiness is detected continuously, an alarm will be triggered.

---

# Dataset Preparation

You can prepare and organize datasets using:

```bash
python create_dataset.py
```

and

```bash
python enumerate_data.py
```

---

# Model Training

To train the model:

```bash
python train_model.py
```

---

# Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- Deep Learning
- Computer Vision

---

# Research Reference

Base Paper:

**A CNN-Based Approach for Driver Drowsiness Detection by Real-Time Eye State Identification**

https://www.mdpi.com/2076-3417/13/13/7849

---

# Future Improvements

- Mobile deployment
- Night vision optimization
- Better low-light detection
- Yawning detection
- Multi-face monitoring

---

# Author

Ayush Mittal(23/CS/106) and Chitranshu Mahaur(23/CS/113)  
DTU (Delhi Technological University)

---

# Repository Link

https://github.com/Chitranshu-DTU-Official/Deep-Learning-Project

---

# Trained Model

The pre-trained model can be downloaded from:

https://drive.google.com/file/d/15uk8tbJVeeJ-IIBgAPDobMEIFmlmV_Xd/view?usp=sharing

Download the model file and place it inside the `models/` directory before running `realtime.py`.
