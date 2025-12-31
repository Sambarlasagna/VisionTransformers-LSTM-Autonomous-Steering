# 🚗 End-to-End Autonomous Steering using Behavioral Cloning

This project implements an end-to-end vision-based autonomous steering system using behavioral cloning. A deep learning model learns to predict continuous steering angles directly from monocular road images by imitating human driving behavior.

The project is inspired by NVIDIA’s end-to-end self-driving car architecture and is implemented entirely in PyTorch.

# 📌 Project Overview

Traditional autonomous driving pipelines split perception, planning, and control into separate modules. In contrast, this project follows an end-to-end learning approach, where a neural network maps raw camera input directly to steering commands.

The model is trained on the Udacity Self-Driving Car Dataset, which consists of front-facing camera images paired with steering angles recorded from a human driver.

# 🧠 Key Features

End-to-end steering angle prediction from raw RGB images

NVIDIA-style CNN architecture for behavioral cloning

Custom image preprocessing and data augmentation

Offline closed-loop video inference for qualitative evaluation

Steering visualization using arrow overlay

Temporal smoothing to reduce steering jitter

# 🗂 Dataset

Source: Udacity Self-Driving Car Simulator Dataset

Input: Front-facing camera images

Label: Continuous steering angle

Only center camera images are used for training and evaluation.

# 🏗 Model Architecture
NVIDIA End-to-End CNN

Convolutional layers for spatial feature extraction

Fully connected layers for regression

ELU activations

Mean Squared Error (MSE) loss

The model directly predicts a real-valued steering angle for each input frame.

# 🧪 Training Details

Framework: PyTorch

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Image preprocessing:

Cropping (remove sky and dashboard)

Color conversion (RGB → YUV)

Gaussian blur

Resizing to (66 × 200)

Normalization

# 🎥 Evaluation Method

1. Quantitative Evaluation

Validation loss (MSE) computed on held-out data

Steering prediction compared with ground truth

2. Qualitative Evaluation (Offline Closed-Loop Inference)

Udacity driving frames are converted into a continuous video

The trained model predicts steering angles frame-by-frame

Steering decisions are visualized using an arrow overlay

Temporal smoothing is applied to improve stability

This evaluation demonstrates how the learned policy would behave if deployed in control.

# 📊 Results

The model successfully learns road curvature and lane-following behavior

Steering predictions are smooth and consistent on straight roads and curves

Offline video inference shows stable autonomous steering decisions

# ⚠️ Limitations

The evaluation is offline (video-based), not real-time closed-loop control

The model relies on imitation learning and inherits biases from human driving data

Performance may degrade under unseen conditions (night driving, heavy traffic)

# 🚀 Future Work

Reintroduce dropout layers for improved generalization

Compare with a ViT + LSTM architecture for temporal modeling

Add steering stability metrics (jerk, variance)

Extend to reinforcement learning or simulation-based control

# 🛠 Tech Stack

- Python

- PyTorch

- OpenCV

- NumPy

- Matplotlib

# 📄 Summary

This project demonstrates a complete end-to-end autonomous steering pipeline, from data preprocessing and model training to evaluation and visualization. It highlights how deep learning models can learn control policies directly from visual input using imitation learning.