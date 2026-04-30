# Attica Wildfire Watch 

An automated, patch-based image classification pipeline built with PyTorch to detect active wildfires and smoke plumes from satellite imagery, specifically focusing on the Attica region of Greece.

##  Project Objective
Early detection of wildfires from space can reduce response times and save lives. This deep learning model is designed to analyze cropped satellite images and perform Binary Classification to determine if a region is currently safe or experiencing a fire anomaly.

##  Repository Structure
* `data/`: (Git-ignored) Contains the training and validation images.
* `src/`: Contains the core mathematical architecture.
  * `dataset.py`: The custom PyTorch Dataset class for loading and transforming images.
  * `model.py`: The Convolutional Neural Network (Transfer Learning via ResNet18).
  * `train.py`: The training loop and loss calculations.

##  Getting Started
1. Clone this repository.
2. Set up a Python virtual environment: `python -m venv venv`
3. Install dependencies: `pip install torch torchvision pillow`
4. Download the [Wildfire Prediction Dataset from Kaggle](https://www.kaggle.com/) and extract it into the `data/train/` directory.


```
--- Starting Epoch 1/2 ---
Batch 0 | Loss: 0.7692
Batch 50 | Loss: 0.2673
Batch 100 | Loss: 0.1842
Batch 150 | Loss: 0.2297
Batch 200 | Loss: 0.1247
Batch 250 | Loss: 0.3274
Batch 300 | Loss: 0.2881
Batch 350 | Loss: 0.2264
Batch 400 | Loss: 0.0761
Batch 450 | Loss: 0.2581
Batch 500 | Loss: 0.2090
Batch 550 | Loss: 0.1571
Batch 600 | Loss: 0.2235
Batch 650 | Loss: 0.1709
Batch 700 | Loss: 0.1710
Batch 750 | Loss: 0.1916
Batch 800 | Loss: 0.2926
Batch 850 | Loss: 0.2231
Batch 900 | Loss: 0.1905
Epoch 1 complete! Final Loss: 0.0537

--- Starting Epoch 2/2 ---
Batch 0 | Loss: 0.1783
Batch 50 | Loss: 0.1734
Batch 100 | Loss: 0.0871
Batch 150 | Loss: 0.1240
Batch 200 | Loss: 0.1221
Batch 250 | Loss: 0.1143
Batch 300 | Loss: 0.2919
Batch 350 | Loss: 0.0623
Batch 400 | Loss: 0.2502
Batch 450 | Loss: 0.2111
Batch 500 | Loss: 0.2744
Batch 550 | Loss: 0.0387
Batch 600 | Loss: 0.1968
Batch 650 | Loss: 0.1953
Batch 700 | Loss: 0.0417
Batch 750 | Loss: 0.2834
Batch 800 | Loss: 0.1225
Batch 850 | Loss: 0.2143
Batch 900 | Loss: 0.0185
Epoch 2 complete! Final Loss: 0.3924
```