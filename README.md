# Handwritten Character Recognition with RNNs and LSTMs

## Overview

This project dives deep into the world of Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks to tackle the challenging task of handwritten character recognition. The primary objective is to classify sequences of 2-dimensional points (representing pen strokes) into five distinct character classes from the Kannada/Telugu script.
    



## Table of Contents
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Results](#models-and-results)
  - [Model Architectures](#model-architectures)
  - [Results](#results)
- [Contributing](#contributing)


## Dataset

The dataset employed in this project contains a curated subset of handwritten characters from the Kannada/Telugu script. Each character is represented as a sequence of 2-dimensional points, consisting of x and y coordinates, marking a single stroke of the character. Notably, the dataset is organized into training and test data, categorized by their respective character folders. Data files store the sequential data points, which have been meticulously preprocessed and normalized to ensure consistency in scale.
We are given the following classes -
    1. a
    2. ai
    3. chA
    4. lA
    5. tA

    1. Train Dataset has 345 files belonging to 5 classes.
    2. Test Dataset has 100 files belonging to 5 classes.

![datasset](https://github.com/Kishan2912/Handwritten-Character-Recognition/assets/83392319/0d8b8918-0446-4517-8885-1f4abd3b32b1)

## Tech Stack

- **Programming Language**: Python

- **Deep Learning Framework**: TensorFlow with Keras
- **Data Manipulation**: NumPy
- **Data Preprocessing**: Scikit-learn
- **Data Visualization**: Matplotlib
- **Normalization**: Scikit-learn's MinMaxScaler

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Kishan2912/Handwritten-Character-Recognition

2. **Go to folder:**
    ```bash
   cd hand_char_recognization

3. **Install dependencies:**
    ```bash
   pip install -r requirements.txt


## Usage

#### Model Training and Evaluation:

Execute the provided Jupyter Notebook or Python scripts to train and assess different RNN and LSTM models.
Modify hyperparameters and model architectures as necessary for experimentation.





#### Analyze Model Performance:

Evaluate model performance, including training and testing accuracy.
Explore model visualizations and confusion matrices to gain insights into model behavior.


## Models and Results
### Model Architectures
This project includes an array of RNN and LSTM model architectures, featuring various cell counts and dense layer dimensions. Some models also incorporate dropout layers for regularization.


![summary of models](https://github.com/Kishan2912/Handwritten-Character-Recognition/assets/83392319/b28f3afe-2ec0-4b79-9878-26271044501a)


### Results
Among the models experimented with, the standout performer achieved an outstanding testing accuracy of 98% using a specific LSTM configuration.

Here's a snapshot of the top-performing models:

![result](https://github.com/Kishan2912/Handwritten-Character-Recognition/assets/83392319/ac3b082e-0229-4675-ac22-53a96824c52e)






## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

