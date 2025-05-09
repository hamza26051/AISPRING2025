import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("customer_data.csv")

df.fillna(df.median(numeric_only=True), inplace=True)
df = df[(np.abs(df.select_dtypes(include=[np.number]) - df.mean()) <= (3 * df.std())).all(axis=1)]

X = df.drop('high_value', axis=1)
y = df['high_value']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

coefficients = model.coef_[0]
intercept = model.intercept_[0]
features = X.columns
print("Separating hyperplane: ", " + ".join([f"{coeff:.2f}*{feat}" for coeff, feat in zip(coefficients, features)]), f"+ ({intercept:.2f}) = 0")

threshold_rule = dict(zip(features, coefficients))
print("Rule-based influence of features:")
for feat, coeff in threshold_rule.items():
    print(f"{feat}: {'increase' if coeff > 0 else 'decrease'} likelihood of being high-value")
