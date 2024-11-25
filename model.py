from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dataset
dataset_path = "meal_plan_dataset.csv"
meal_dataset_path = "meal_dataset.csv"

data = pd.read_csv(dataset_path)
meal_data = pd.read_csv(meal_dataset_path)

data = data.dropna()
meal_data = meal_data.dropna()

# Normalize data
X = data[['age', 'weight', 'height']].values
y = data[['breakfast_calories', 'lunch_calories', 'dinner_calories']].values

X_min, X_max = X.min(axis=0), X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Input(shape=(3,)),  # Use Input to define the shape of the input data
    Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(3)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test), verbose=0)

# API Endpoints
@app.route('/')
def home():
    return "Meal Plan API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.json
        age = user_input.get('age')
        weight = user_input.get('weight')
        height = user_input.get('height')
        if age is None or weight is None or height is None:
            return jsonify({"error": "Missing input fields (age, weight, height)"}), 400

        # Normalize input
        new_user = np.array([[age, weight, height]])
        new_user = (new_user - X_min) / (X_max - X_min)

        # Predict
        predicted_calories = model.predict(new_user)
        return jsonify({
            "breakfast_calories": predicted_calories[0][0],
            "lunch_calories": predicted_calories[0][1],
            "dinner_calories": predicted_calories[0][2]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-meal-plan', methods=['POST'])
def generate_meal_plan():
    try:
        user_input = request.json
        age = user_input.get('age')
        weight = user_input.get('weight')
        height = user_input.get('height')
        if age is None or weight is None or height is None:
            return jsonify({"error": "Missing input fields (age, weight, height)"}), 400

        # Normalize input
        new_user = np.array([[age, weight, height]])
        new_user = (new_user - X_min) / (X_max - X_min)

        # Predict calories
        predicted_calories = model.predict(new_user)

        # Generate meal plan
        def generate_weekly_meal_plan(predicted_calories, meal_data):
            weekly_meal_plan = []
            used_meals = set()
            days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            for day_name in days_of_week:
                daily_meal_plan = []
                for i, meal_type in enumerate(["Breakfast", "Lunch", "Dinner"]):
                    filtered_meals = meal_data[(meal_data["type"] == meal_type) & (~meal_data["title"].isin(used_meals))]

                    if filtered_meals.empty:
                        filtered_meals = meal_data[meal_data["type"] == meal_type]

                    best_match = filtered_meals.iloc[
                        (filtered_meals["calories"] - predicted_calories[0][i]).abs().argsort()[:1]
                    ]

                    for _, meal in best_match.iterrows():
                        used_meals.add(meal["title"])
                        daily_meal_plan.append({
                            "type": meal["type"],
                            "title": meal["title"],
                            "calories": int(meal["calories"]),
                            "macros": {
                                "protein": int(meal["protein"]),
                                "carbs": int(meal["carbs"]),
                                "fat": int(meal["fat"])
                            },
                            "ingredients": meal["ingredients"].split(", "),
                            "recipe": meal["recipe"]
                        })

                weekly_meal_plan.append({"day": day_name, "meals": daily_meal_plan})

            return weekly_meal_plan

        weekly_meal_plan = generate_weekly_meal_plan(predicted_calories, meal_data)
        return jsonify(weekly_meal_plan)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test-model', methods=['GET'])
def test_model():
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return jsonify({"r2_score": r2})

# Run the app
if __name__ == '__main__':
    # Use dynamic port if available, default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
