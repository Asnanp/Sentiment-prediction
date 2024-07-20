# Sentiment Analysis Project

## Overview

This repository contains a comprehensive sentiment analysis project using both Logistic Regression and Neural Networks. The project demonstrates various techniques for processing and analyzing text data, including preprocessing, vectorization, model training, evaluation, and visualization.

## Features

- **Data Preprocessing**: Includes cleaning and tokenizing text data.
- **Feature Extraction**: Uses TF-IDF and other vectorization techniques to convert text data into numerical features.
- **Model Training**: Implements Logistic Regression and Neural Network models for sentiment classification.
- **Evaluation**: Provides detailed metrics and confusion matrix visualizations to evaluate model performance.
- **Prediction**: Allows prediction of sentiment for new text samples.

## Installation

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Asnanp/Sentiment-prediction.git
cd Sentiment-prediction
pip install -r requirements.txt
```
**Usage**
Prepare Data: Ensure your dataset is formatted correctly and loaded into the df DataFrame.

Train Models:
For Logistic Regression:
```from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
```
For Neural Network:

```from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
nn_model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_len),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(num_classes, activation='softmax')
])
nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
```

Evaluate Models:
Generate confusion matrices and classification reports.
Visualize using Plotly and Matplotlib:
```
import plotly.graph_objects as go
```
# Create and display confusion matrix plots
Results
Logistic Regression: Achieved an accuracy of X% with detailed confusion matrix and classification report available.
Neural Network: Achieved an accuracy of Y% with performance metrics and visualizations provided.
Visualizations
Interactive and static visualizations of confusion matrices and performance metrics are included in the repository. Refer to visualizations/ directory for more details.

Contributing
Contributions to this project are welcome! Please fork the repository and submit a pull request with your improvements or fixes.


Acknowledgments
Libraries: Scikit-learn, TensorFlow, Plotly, NLTK, Pandas, NumPy

Dataset: https://www.kaggle.com/code/mesutssmn/sentiment-analysis-for-mental-health
