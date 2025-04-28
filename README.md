# LSTM-based Viewport Prediction in VR Streaming

## Overview

This project focuses on predicting VR viewport tile activations using sensory data (orientation and raw sensor readings). The solution leverages **data augmentation**, **sequential modeling with LSTM**, and comparisons with traditional machine learning classifiers like **Logistic Regression** and **Linear SVC**.

The goal is to create a robust model that can predict a user's probable viewport, enabling **efficient streaming** of only the relevant parts of a VR scene, thereby optimizing **bandwidth** and **rendering resources**.

---

## Project Structure

### **STEP 1: Data Preprocessing**
- Augments orientation and raw data using Gaussian noise to simulate real-world variability.
- Ensures consistent frame numbering and timestamp updating.
- Generates tile mappings using KNN nearest neighbor matching on augmented orientation data.

### **STEP 2: Algorithm Training**
- Constructs sequential datasets using a sequence length of 10 frames.
- Normalizes the features and applies multi-label binarization to the tile outputs.
- Trains an LSTM-based model using PyTorch to predict multi-label tile activations.
- Evaluates the model using **Precision**, **Recall**, **F1 Score**, and **IoU (Jaccard Index)**.
- Visualizes model performance and sample predictions.

### **STEP 3: Model Comparison**
- Trains fast baseline models (Logistic Regression, Linear SVC) on a random 20% sample.
- Compares performance against the LSTM model using consistent evaluation metrics.
- Provides bar chart visualizations and tabular summaries of model performances.

---

## Key Features
- Data augmentation for better generalization.
- Sequential input modeling using LSTM networks.
- Multi-label classification approach for viewport prediction.
- Robust model evaluation based on Precision, Recall, F1 Score, and IoU.
- Comparative study between deep learning and traditional machine learning models.
- Detailed visual analysis including boxplots, histograms, heatmaps, and metric trend plots.

---

## Installation

Install the required Python libraries using:

```bash
pip install torch scikit-learn pandas tqdm matplotlib seaborn
```

Ensure the dataset folders are properly organized as:

```
/content/cleaned_data/sensory/orientation
/content/cleaned_data/sensory/raw
/content/cleaned_data/sensory/tile
```

The augmented and generated outputs will be stored in:

```
/content/output_augmented
/content/output_aug_tiles
```

---

## How to Run

1. **Data Augmentation and Tile Generation:**
   - Execute the preprocessing scripts to create augmented orientation, raw, and tile data.

2. **Training the LSTM Model:**
   - Run the training script to build and evaluate the LSTM-based viewport prediction model.

3. **Model Comparison:**
   - Load the preprocessed data and saved LSTM metrics.
   - Train Logistic Regression and Linear SVC on sampled data.
   - Compare the traditional models with the LSTM model using evaluation metrics and visualizations.

---

## Results

- **LSTM significantly outperforms traditional models** (Logistic Regression, Linear SVC) on Precision, Recall, F1 Score, and IoU.
- Highlights the advantage of **sequence-based modeling** for VR viewport prediction.
- Visualization plots illustrate training progress, feature distributions, and sample-level model performance.

---

## Future Work

- Extend the dataset with more diverse user head movements.
- Implement lightweight transformer-based models for real-time VR streaming.
- Optimize model deployment on VR headsets for real-time, on-device prediction.

---

## Credits

Developed as part of a **VR Viewport Prediction Pipeline** project using **data-driven AI modeling techniques**.
