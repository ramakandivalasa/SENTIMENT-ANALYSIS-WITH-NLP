# SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: RAMA KANDIVALASA 

*INTERN ID*: CT04DN904

*DOMAIN*: MACHINE LEARNING 

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

## 📌 Project Title: PERFORM SENTIMENT ANALYSIS ON A DATASET OF CUSTOMER REVIEWS USING TF-IDF VECTORIZATION AND LOGISTIC REGRESSION

## 🔍 Objective
Perform sentiment classification on airline-related tweets using Natural Language Processing (NLP) techniques. This project uses **TF-IDF Vectorization** and a **Logistic Regression** classifier to identify whether a tweet expresses a **positive**, **neutral**, or **negative** sentiment.

---

## 📊 Dataset
- **Source**: Twitter US Airline Sentiment Dataset
- **Columns Used**:
  - `text`: Tweet content
  - `airline_sentiment`: Labeled sentiment (positive/neutral/negative)

---

## 🧪 Process Overview

1. **Data Preprocessing**
   - Lowercasing
   - Removing URLs, mentions, hashtags, punctuation
   - Stopword removal
2. **TF-IDF Vectorization**
3. **Model Training**
   - Logistic Regression
4. **Evaluation**
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion matrix

---

## 🛠 Technologies
- Python
- Pandas, NLTK
- Scikit-learn
- Matplotlib / Seaborn

---

## 📈 Results
The model performs multi-class classification with reasonable accuracy. Further tuning or deep learning could improve results.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
jupyter notebook sentiment_analysis.ipynb
exit
---

# **Output**

text airline_sentiment
0                @VirginAmerica What @dhepburn said.           neutral
1  @VirginAmerica plus you've added commercials t...          positive
2  @VirginAmerica I didn't today... Must mean I n...           neutral
3  @VirginAmerica it's really aggressive to blast...          negative
4  @VirginAmerica and it's a really big bad thing...          negative
Confusion Matrix:
 [[1769   79   41]
 [ 291  244   45]
 [ 139   44  276]]

Classification Report:
               precision    recall  f1-score   support

    negative       0.80      0.94      0.87      1889
     neutral       0.66      0.42      0.52       580
    positive       0.76      0.60      0.67       459

    accuracy                           0.78      2928
   macro avg       0.74      0.65      0.68      2928
weighted avg       0.77      0.78      0.77      2928


