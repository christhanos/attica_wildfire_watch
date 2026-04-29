# Attica Wildfire Watch 🛰️🔥

An automated, patch-based image classification pipeline built with PyTorch to detect active wildfires and smoke plumes from satellite imagery, specifically focusing on the Attica region of Greece.

## 🎯 Project Objective
Early detection of wildfires from space can reduce response times and save lives. This deep learning model is designed to analyze cropped satellite images and perform Binary Classification to determine if a region is currently safe or experiencing a fire anomaly.

## 🗂️ Repository Structure
* `data/`: (Git-ignored) Contains the training and validation images.
* `src/`: Contains the core mathematical architecture.
  * `dataset.py`: The custom PyTorch Dataset class for loading and transforming images.
  * `model.py`: The Convolutional Neural Network (Transfer Learning via ResNet18).
  * `train.py`: The training loop and loss calculations.

## 🚀 Getting Started
1. Clone this repository.
2. Set up a Python virtual environment: `python -m venv venv`
3. Install dependencies: `pip install torch torchvision pillow`
4. Download the [Wildfire Prediction Dataset from Kaggle](https://www.kaggle.com/) and extract it into the `data/train/` directory.