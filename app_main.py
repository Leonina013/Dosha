import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Function to get Pitta category and nutrition advice
def get_pitta_category(pitta_score):
    if pitta_score <= 5:
        return "No to Light Pitha"
    elif pitta_score <= 8:
        return "Moderate Pitha"
    else:
        return "Extreme Pitha"

def get_pitta_nutrition_advice(pitta_category):
    # Nutrition advice for each category
    nutrition_advice = {
        "No to Light Pitha": "Favor cooling foods...",
        "Moderate Pitha": "Balanced meals with cooling and slightly warming foods...",
        "Extreme Pitha": "Emphasize cooling and calming foods..."
    }
    return nutrition_advice.get(pitta_category, "")

# Function to predict Pitta score
def predict_pitta_score(input_values):
    pitta_feature_columns = ['AverageHeartRate', 'CumulativeSteps', 'ActiveDistance', 'LightActiveDistance', 'MinutesAsleep', 'Calories']
    pitta_scaler = StandardScaler()

    pitta_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    pitta_dataset = pd.read_csv('Pitha_Dataset.csv')
    X_pitta = pitta_dataset[pitta_feature_columns]
    y_pitta = pitta_dataset['Pitha_Score']

    X_pitta_train, _, y_pitta_train, _ = train_test_split(X_pitta, y_pitta, test_size=0.2, random_state=42)

    X_pitta_train_scaled = pitta_scaler.fit_transform(X_pitta_train)

    pitta_model.fit(X_pitta_train_scaled, y_pitta_train)

    input_values_df = pd.DataFrame([input_values])
    input_values_scaled = pitta_scaler.transform(input_values_df)

    predicted_pitta_score = pitta_model.predict(input_values_scaled)
    pitta_category = get_pitta_category(predicted_pitta_score[0])
    nutrition_advice = get_pitta_nutrition_advice(pitta_category)

    return predicted_pitta_score[0], pitta_category, nutrition_advice

# Function to predict Vata score
def predict_vata_score(input_values):
    vata_feature_columns = ['TotalMinutesAsleep', 'BedtimeRoutine', 'SleepQuality', 'TotalSteps', 'SedentaryMinutes', 'ModeratelyActiveMinutes']

    vata_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)

    vata_dataset = pd.read_csv('Vata_Dataset_with_Scores.csv')
    X_vata = vata_dataset[vata_feature_columns]
    y_vata = vata_dataset['Vata_Score']

    X_vata_train, _, y_vata_train, _ = train_test_split(X_vata, y_vata, test_size=0.2, random_state=42)

    vata_model.fit(X_vata_train, y_vata_train)

    input_values_df = pd.DataFrame([input_values])

    predicted_vata_score = vata_model.predict(input_values_df)
    vata_category = "No to Light Vata" if predicted_vata_score[0] <= 5 else ("Moderate Vata" if predicted_vata_score[0] <= 8 else "Extreme Vata")

    return predicted_vata_score[0], vata_category

# Streamlit app
st.title("Dosha Score Prediction")
st.sidebar.title("Enter Dosha Features")

# Sidebar inputs
pitta_input_values = {
    'AverageHeartRate': st.sidebar.number_input("Average Heart Rate"),
    'CumulativeSteps': st.sidebar.number_input("Cumulative Steps"),
    'ActiveDistance': st.sidebar.number_input("Active Distance"),
    'LightActiveDistance': st.sidebar.number_input("Light Active Distance"),
    'MinutesAsleep': st.sidebar.number_input("Minutes Asleep"),
    'Calories': st.sidebar.number_input("Calories")
}

vata_input_values = {
    'TotalMinutesAsleep': st.sidebar.number_input("Total Minutes Asleep (Vata)"),
    'BedtimeRoutine': st.sidebar.number_input("Bedtime Routine (Vata)"),
    'SleepQuality': st.sidebar.number_input("Sleep Quality (Vata)"),
    'TotalSteps': st.sidebar.number_input("Total Steps (Vata)"),
    'SedentaryMinutes': st.sidebar.number_input("Sedentary Minutes (Vata)"),
    'ModeratelyActiveMinutes': st.sidebar.number_input("Moderately Active Minutes (Vata)")
}

# Predict scores
predicted_pitta_score, pitta_category, pitta_nutrition_advice = predict_pitta_score(pitta_input_values)
predicted_vata_score, vata_category = predict_vata_score(vata_input_values)

# Display results
st.write("## Pitta Dosha")
st.write("Predicted Pitta Score:", predicted_pitta_score)
st.write("Predicted Pitta Category:", pitta_category)
st.write("Nutrition Advice for Pitta:", pitta_nutrition_advice)

st.write("## Vata Dosha")
st.write("Predicted Vata Score:", predicted_vata_score)
st.write("Predicted Vata Category:", vata_category)
