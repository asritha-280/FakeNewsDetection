import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load your CSV files
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Add labels
df_fake['label'] = 'FAKE'
df_true['label'] = 'REAL'

# Combine datasets
df = pd.concat([df_fake, df_true], ignore_index=True)
df = df[['text', 'label']]
df = df.sample(frac=1).reset_index(drop=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test_vec))
print(f"Model Accuracy: {round(acc * 100, 2)}%")

# Save model and vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
