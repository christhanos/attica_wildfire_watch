# 🔥 Attica Wildfire Watch: AI Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wildfiredetectionw.streamlit.app/)

An AI-powered satellite imagery analysis tool built to detect early-stage wildfire smoke plumes in the Attica, Greece region. 

## Project Overview
This project utilizes a Convolutional Neural Network (ResNet18) employing Transfer Learning to classify high-resolution satellite imagery. The model was trained to distinguish between safe forest topologies and active wildfire events.

## Results
* **Training Architecture:** ResNet18
* **Training Epochs:** 2
* **Final Academic Accuracy:** 96.21% (6,061 / 6,300 unseen images)

## Limitations & Domain Shift Discovery
During rigorous edge-case testing, the model exhibited vulnerabilities to **Domain Shift** (Out-of-Distribution data). Specifically, the model produced False Positives when processing:
1. Highly reflective local topography (e.g., Mount Penteli marble quarries).
2. Dense, low-hanging winter cloud cover.
3. Superimposed UI elements (map pins, text overlays).

**Future Work:** To improve model robustness and mitigate these False Positives, the next iteration will implement **Hard Negative Mining**. By compiling a custom dataset of Attica's specific reflective topography and cloud systems—explicitly labeled as 'Safe'—the convolutional filters can be optimized to better differentiate between geological formations and turbulent smoke plumes.


