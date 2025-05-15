# DistilBERT Fine-tuning Tutorial

This tutorial provides a step-by-step guide to fine-tuning a DistilBERT model for sentiment analysis on the IMDb movie reviews dataset. The code is structured in a modular way, breaking down the entire process into clear, separate components that students can understand and modify.

## Overview

This tutorial covers:
1. **Dataset Exploration**: Understanding the IMDb dataset
2. **Data Preparation**: Tokenizing and preparing the data for training
3. **Model Training**: Fine-tuning the DistilBERT model
4. **Model Evaluation**: Evaluating model performance on the test set
5. **Result Visualization**: Visualizing performance metrics 
6. **Inference**: Making predictions with the fine-tuned model

## Requirements

To run this tutorial, you'll need:

```python
transformers>=4.20.0
datasets>=2.6.0
torch>=1.10.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
numpy>=1.20.0