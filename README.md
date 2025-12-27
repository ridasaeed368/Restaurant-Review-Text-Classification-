🍽️ Restaurant Review Text Classification (NLP)

This project implements text classification on restaurant reviews using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to classify customer reviews as positive or negative based on their textual content.

✨ Key Features

Text preprocessing and cleaning of restaurant reviews

Removal of special characters and stopwords

Text normalization using stemming

Bag of Words (BoW) feature representation

Classification using Machine Learning models

Performance evaluation using accuracy and confusion matrix

🧠 Methodology
🔹 Text Preprocessing

Removal of non-alphabetic characters

Conversion to lowercase

Tokenization

Stopword removal using NLTK

Porter Stemming

🔹 Feature Extraction

Bag of Words (BoW) using CountVectorizer

Maximum feature size set to improve performance

🔹 Classification Models

Logistic Regression (implemented)

Support Vector Machine (SVM) – tested

Naive Bayes (Gaussian & Multinomial) – tested

K-Nearest Neighbors (KNN) – tested

🛠️ Technologies Used

Python

Pandas

NumPy

NLTK

Scikit-learn

Matplotlib

📂 Dataset

File: Restaurant_Reviews.tsv

Format: Tab-separated values

Columns:

Review – customer restaurant review text

Liked – target label (positive / negative)

📊 Model Evaluation

Train-test split (80% training, 20% testing)

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Accuracy score comparison across models

🎓 Purpose

This project is designed for educational and academic purposes, demonstrating how NLP techniques can be applied to sentiment analysis and text classification of restaurant reviews.

⚠️ Note

This is a classical NLP approach using Bag of Words

Deep learning models are not used

Suitable for beginners learning NLP workflows

👩‍💻 Author

Rida Saeed
