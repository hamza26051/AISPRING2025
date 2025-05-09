import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv("emails.csv")

text_cols = ['sender_address']
for col in text_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('is_spam', axis=1)
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(le, "label_encoder.pkl")

new_email = {
    'word_freq_offer': 0.5,
    'word_freq_money': 0.3,
    'word_freq_free': 0.1,
    'email_length': 150,
    'num_hyperlinks': 2,
    'sender_address': 'promo@example.com'
}
new_email_df = pd.DataFrame([new_email])
new_email_df['sender_address'] = joblib.load("label_encoder.pkl").transform(new_email_df['sender_address'])

loaded_model = joblib.load("spam_classifier_model.pkl")
prediction = loaded_model.predict(new_email_df)
print("Spam Prediction (1 = spam, 0 = not spam):", prediction[0])
