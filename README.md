# project_5.Sentiment-Analysis
* AI Echo: Your Smartest Conversational Partner
This project performs multi-class sentiment analysis on a ChatGPT review dataset using both traditional machine learning algorithms and advanced deep learning models. The objective is to classify user reviews into multiple sentiment categories with high predictive performance.
📌 Problem Statement:
Given user-generated ChatGPT reviews, build a robust classification system capable of accurately predicting sentiment categories using multiple modeling approaches and compare their performance.
📊 Models Implemented:
🔹 Traditional Machine Learning:
Logistic Regression
Random Forest Classifier
Naive Bayes
🔹 Deep Learning
LSTM (Long Short-Term Memory Network)
🔹 Transformer-Based Model
BERT (Bidirectional Encoder Representations from Transformers)
Model Comparison Strategy
Text preprocessing and cleaning
TF-IDF
Tokenization & Padding (for LSTM)
Pretrained Transformer embeddings (for BERT)
Performance evaluation using:
Accuracy
Precision
Recall
F1-score
Results:
All models achieved 100% classification accuracy on the evaluation dataset.
Technologies Used
Python
Scikit-learn
TensorFlow / Keras
PyTorch / Transformers
Pandas
NumPy
Matplotlib / Seaborn
Streamlit Deployment:
The final model was deployed using Streamlit to create an interactive web interface where users can:
Enter custom review text
Get real-time sentiment predictions    
View prediction confidence scores
