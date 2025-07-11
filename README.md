# 📰 Fake News Detection using Machine Learning & Deep Learning

This project focuses on building a robust pipeline to classify news articles as **Fake** or **Real** using various Machine Learning and Deep Learning techniques. The dataset used for this project is sourced from [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

---

## 📂 Project Structure

- `data/`: Dataset loading and preprocessing
- `models/`: Training models (ML & DL)
- `evaluation/`: Model comparison and performance metrics
- `visualizations/`: Charts and plots
- `output/`: Cleaned data and results

---

## 🔧 Technologies Used

- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- NLTK (text preprocessing)
- TensorFlow / Keras (ANN)
- Google Colab (for development)

---

## 📌 Dataset Overview

- **Source**: Kaggle — Fake and Real News Dataset
- **Files**: `Fake.csv` and `True.csv`
- **Shape after cleaning**: 38,646 rows × 3 columns
- **Features**: News article `text` and its `label` (fake or real)

---

## 🔄 Data Preprocessing Steps

- Concatenated fake and real news
- Shuffled and labeled data
- Removed duplicates
- Preprocessed text:
  - Lowercased, removed special characters and numbers
  - Removed stopwords
  - Applied stemming
- Transformed text to TF-IDF vectors

---

## 🧠 Models Implemented

| Model                    | Accuracy | Precision | Recall | F1-Score | AUC   |
|-------------------------|----------|-----------|--------|----------|--------|
| Logistic Regression      | 98.14%   | 97.81%    | 98.82% | 98.31%   | 0.9970 |
| Decision Tree            | 99.35%   | 99.34%    | 99.48% | 99.41%   | 0.9934 |
| Random Forest            | 98.90%   | 98.57%    | 99.43% | 99.00%   | 0.9990 |
| K-Nearest Neighbors      | 86.49%   | 90.21%    | 84.55% | 87.29%   | 0.9185 |
| Support Vector Machine   | 97.49%   | 97.35%    | 98.09% | 97.72%   | 0.9963 |
| Naive Bayes              | 92.68%   | 93.26%    | 93.39% | 93.33%   | 0.9761 |
| Artificial Neural Network| 98.20%   | 98.10%    | 98.63% | 98.36%   | 0.9978 |

---

## 📈 Visualizations

- Class distribution bar chart
- Word cloud of most frequent words
- Heatmap of feature correlations
- Confusion matrices & ROC curves per model
- Final model comparison chart

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
