# RNN Implementation for Spam Email Classification

## Overview
This repository contains an implementation of a **Recurrent Neural Network (RNN) using LSTMs** for text classification. The model is built using **TensorFlow/Keras** and performs text preprocessing using **NLTK and Pandas**.

## Features
- **Data Preprocessing:** Cleans and tokenizes text data.
- **Model Architecture:** Uses an LSTM-based neural network.
- **Dataset Handling:** Reads data from `combined_data.csv`.
- **Training & Evaluation:** Trains and evaluates the model on labeled text data.

## Requirements
Install the necessary dependencies:
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/RNN_Implementation.git
   cd RNN_Implementation
   ```
2. Place your dataset (`combined_data.csv`) in the working directory.
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook RNN_implementation.ipynb
   ```

## Model Architecture
- **Embedding Layer**: Converts text to dense vectors.
- **LSTM Layer**: Captures sequential dependencies.
- **Dense Layer**: Outputs classification results.

## Dataset
Ensure that your dataset (`combined_data.csv`) has at least the following columns:
- `text`: The text data.
- `label`: The corresponding class labels.

## Results
- The notebook includes code for **accuracy evaluation** and **visualization of results**.
- You can modify hyperparameters to improve performance.

## Author
Ayesha Jabeen  
Feel free to contribute or open an issue!

### Dataset link: 
Spam Email Classification: https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset

