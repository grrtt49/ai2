import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import LabelEncoder #for feature engineering 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error # for␣evaluating ml models
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv")
data = pd.get_dummies(data, columns=['region'], prefix='region', dtype=int)

label_encoder = LabelEncoder()
data['smoker_encoded'] = label_encoder.fit_transform(data['smoker'])

data['sex_encoded'] = label_encoder.fit_transform(data['sex'])

data = data[[x for x in data.columns if x not in ['smoker', 'sex']]]

# Select the relevant features
X = data[['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded']]
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_predictions)
gb_mae = mean_absolute_error(y_test, gb_predictions)

# Show how well the model works
# plt.figure(figsize=(8, 4))
# plt.scatter(y_test, gb_predictions, color='purple', label='Gradient Boosting')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
# plt.title('Gradient Boosting: Actual vs. Predicted') 
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.legend()
# plt.show()


# Function to predict charges
def predict_charges():
    age = float(age_entry.get())
    sex = sex_combobox.get()
    bmi = float(bmi_entry.get())
    children = int(children_entry.get())
    smoker = smoker_combobox.get()

    sex_encoded = 0 if sex == 'female' else 1
    smoker_encoded = 1 if smoker == 'yes' else 0

    input_data = [[age, sex_encoded, bmi, children, smoker_encoded]]
    predicted_charge = gb_model.predict(input_data)[0]

    result_label.config(text=f'Predicted Charges: ${predicted_charge:.2f}')

# Create the main window
window = tk.Tk()
window.title("Insurance Charges Predictor")

# Labels
age_label = ttk.Label(window, text="Age:")
bmi_label = ttk.Label(window, text="BMI:")
children_label = ttk.Label(window, text="Number of Children:")
sex_label = ttk.Label(window, text="Sex:")
smoker_label = ttk.Label(window, text="Smoker:")
result_label = ttk.Label(window, text="Predicted Charges:")

# Entry fields
age_entry = ttk.Entry(window)
bmi_entry = ttk.Entry(window)
children_entry = ttk.Entry(window)

# Comboboxes
sex_combobox = ttk.Combobox(window, values=["male", "female"])
sex_combobox.set("male")  # Default value
smoker_combobox = ttk.Combobox(window, values=["yes", "no"])
smoker_combobox.set("no")  # Default value

# Predict button
predict_button = ttk.Button(window, text="Predict", command=predict_charges)

# Layout
age_label.grid(row=0, column=0)
age_entry.grid(row=0, column=1)
sex_label.grid(row=1, column=0)
sex_combobox.grid(row=1, column=1)
bmi_label.grid(row=2, column=0)
bmi_entry.grid(row=2, column=1)
children_label.grid(row=3, column=0)
children_entry.grid(row=3, column=1)
smoker_label.grid(row=4, column=0)
smoker_combobox.grid(row=4, column=1)
predict_button.grid(row=5, columnspan=2)
result_label.grid(row=6, columnspan=2)

window.mainloop()
