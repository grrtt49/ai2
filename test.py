import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the CSV data
data = pd.read_csv('claims_data.csv')

# Convert Date of Birth and Date of Service to datetime objects
data['DOB'] = pd.to_datetime(data['DOB'])
data['DOS'] = pd.to_datetime(data['DOS'])

# Encode categorical variables like Sex using one-hot encoding
data = pd.get_dummies(data, columns=['SEX'])

# You may also want to extract features from the date, such as year, month, and day
data['DOB_year'] = data['DOB'].dt.year
data['DOB_month'] = data['DOB'].dt.month
data['DOB_day'] = data['DOB'].dt.day
data['DOS_year'] = data['DOS'].dt.year
data['DOS_month'] = data['DOS'].dt.month
data['DOS_day'] = data['DOS'].dt.day

# Define the input features and target variable
X = data[['DOB_year', 'DOB_month', 'DOB_day', 'DOS_year', 'DOS_month', 'DOS_day', 'SEX_F', 'SEX_M']]
y = data['CLAIM']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification, so use 'sigmoid' activation
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

X_new = pd.DataFrame({
    'DOB_year': [1990, 1985, 2000],
    'DOB_month': [5, 8, 2],
    'DOB_day': [15, 20, 10],
    'DOS_year': [2023, 2023, 2023],
    'DOS_month': [4, 7, 1],
    'DOS_day': [10, 25, 5],
    'SEX_F': [1, 0, 1],
    'SEX_M': [0, 1, 0]
})

# Make predictions on a new dataset (X_new)
predictions = model.predict(X_new)

# You can set a threshold to classify claims as fraud or not
threshold = 0.5
fraudulent_claims = (predictions > threshold).astype(int)

print(predictions)
print(fraudulent_claims)
