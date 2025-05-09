import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("house_data.csv")  
print("Initial data:")
print(df.head())

for col in ['square_feet', 'bedrooms', 'bathrooms', 'age']:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

if df['neighborhood'].isnull().sum() > 0:
    df['neighborhood'].fillna(df['neighborhood'].mode()[0], inplace=True)

df = pd.get_dummies(df, columns=['neighborhood'], drop_first=True)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

print("\nEncoded feature columns:")
print(X.columns)

new_house = {
    'square_feet': 2000,
    'bedrooms': 3,
    'bathrooms': 2,
    'age': 10,
}

for col in X.columns:
    if 'neighborhood_' in col:
        new_house[col] = 0

new_house['neighborhood_Suburb'] = 1  

new_df = pd.DataFrame([new_house])
predicted_price = model.predict(new_df)[0]

print(f"\nPredicted price for the new house: ${predicted_price:,.2f}")
