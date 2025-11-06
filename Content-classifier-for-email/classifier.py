import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- Step 1: Load dataset safely ---
try:
    df = pd.read_csv("dataset.csv", encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv("dataset.csv", encoding='latin1')

# --- Step 2: Handle extra columns ---
# If dataset has more than 2 columns, keep only text + label columns
if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
elif 'label' in df.columns and 'message' in df.columns:
    df = df[['label', 'message']]
else:
    # Try to automatically pick first two columns
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']

# --- Step 3: Encode labels ---
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# --- Step 4: Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# --- Step 5: Text vectorization ---
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Step 6: Train SVM model ---
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# --- Step 7: Evaluate ---
y_pred = svm_model.predict(X_test_tfidf)
print("✅ Model trained successfully!")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Step 8: Save model and vectorizer ---
pickle.dump(svm_model, open("svm_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("\n💾 Model and vectorizer saved as 'svm_model.pkl' and 'vectorizer.pkl'")
