import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data_path = 'heart_rate_scores.csv'
df = pd.read_csv(data_path)

# Split the dataset into features (X) and target (y)
X = df[['Value']]  # Removed 'Id' from the features
y = df['Score']

# Initialize the RandomForestClassifier (you can choose other classifiers as well)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the entire dataset
clf.fit(X, y)

def predict_dosha_predominance(heartbeat_value):
    # Predict Dosha predominance using the trained classifier
    predicted_scores = clf.predict([[heartbeat_value]])

    # Map the predicted scores to Dosha types
    dosha_mapping = {1: 'Vata Predominant', 2: 'Pitta Predominant', 3: 'Kapha Predominant'}

    # Get the predicted Dosha type
    predicted_dosha = dosha_mapping.get(predicted_scores[0], 'Abnormal Value of Resting Heart Rate')

    return predicted_dosha




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
        "No to Light Pitha": '''\n
        Favor cooling foods: Incorporate foods that have a cooling effect on the body.\n
        •Include sweet, bitter, and astringent tastes: Focus on foods with these tastes to balance excess heat.\n
        •Limit spicy, oily, and acidic foods: Reduce or avoid foods that can increase Pitta, such as hot spices, fried foods, and excessive sourness.\n
        •Stay hydrated: Drink cool or room temperature water and herbal teas to balance the heat.\n
        •Enjoy fresh, ripe, and sweet fruits: Opt for sweet fruits like melons, grapes, and sweet berries.\n
        Sample No to Light Pitta Diet:\n
        - Breakfast: A bowl of cool oatmeal with ripe berries.\n
        - Lunch: Steamed vegetables with quinoa and a cooling mint-cucumber yogurt sauce.\n
        - Snack: Sliced watermelon or a cool cucumber salad.\n
        - Dinner: Basmati rice with steamed zucchini and a small serving of sweet and juicy fruits.\n
        ''',
        "Moderate Pitha": '''\n
        Balanced meals: Include a mix of cooling and slightly warming foods to maintain equilibrium.\n
        •Enjoy a variety of tastes: Incorporate sweet, bitter, astringent, and a moderate amount of pungent tastes.\n
        •Moderation in spices: Use milder spices like coriander, fennel, and cardamom in your meals.\n
        •Stay hydrated: Drink room temperature water and herbal teas.\n
        •Include fresh and cooked foods: Combine raw and cooked vegetables, grains, and legumes.\n
        Sample Moderate Pitta Diet:\n
        - Breakfast: A fruit smoothie with ripe bananas, mangoes, and a pinch of cardamom.\n
        - Lunch: Mixed greens salad with roasted vegetables, quinoa, and a lemon-olive oil dressing.\n
        - Snack: Sliced apples with almond butter.\n
        - Dinner: Baked salmon with steamed asparagus and a side of basmati rice.\n
        ''',
        "Extreme Pitha": '''\n
        Emphasize cooling and calming foods: Prioritize foods with a strong cooling effect.\n
        •Favor sweet, bitter, and astringent tastes: These tastes help balance excess heat and acidity.\n
        •Avoid hot and spicy foods: Steer clear of very spicy, oily, and fried foods.\n
        •Stay well-hydrated: Drink cool water, coconut water, and herbal teas.\n
        •Include plenty of fresh fruits: Opt for sweet, juicy fruits that help cool the body.\n
        Sample High Pitta Diet:\n
        - Breakfast: A bowl of cool, cooked barley cereal with sliced peaches.\n
        - Lunch: Cucumber and mint raita with rice or quinoa and a side of steamed broccoli.\n
        - Snack: A handful of sweet grapes.\n
        - Dinner: Baked or steamed white fish with a side of lightly steamed carrots and zucchini.\n
        '''
    }
    return nutrition_advice.get(pitta_category, "")


def get_vata_nutrition_advice(vata_category):
    nutrition_advice = {
        "No to Light Vata": '''\n
        •Favor warm, cooked, and easily digestible foods.\n
        •Incorporate plenty of healthy fats such as ghee, coconut oil, and olive oil.\n
        •Include nourishing and grounding foods like sweet potatoes, whole grains (cooked), cooked vegetables, and lentils.\n
        •Drink warm herbal teas (non-caffeinated) like ginger, cinnamon, and licorice.\n
        •Reduce raw foods, cold foods, and excessive caffeine.''',
        "Moderate Vata": '''\n
        •Favor warm, cooked, and easily digestible foods.\n
        •Incorporate plenty of healthy fats such as ghee, coconut oil, and olive oil.\n
        •Include nourishing and grounding foods like sweet potatoes, whole grains (cooked), cooked vegetables, and lentils.\n
        •Drink warm herbal teas (non-caffeinated) like ginger, cinnamon, and licorice.\n
        •Reduce raw foods, cold foods, and excessive caffeine.\n
        •Include a variety of cooked vegetables, grains, and legumes.\n
        •Incorporate small amounts of dairy, if tolerated (e.g., warm milk with spices).\n
        •Include foods with mild natural sweetness like ripe fruits (in moderation) and sweet spices.\n
        •Hydrate well with warm water, herbal teas, and warm soups.''',
        "Extreme Vata": '''\n
        •Focus on stabilizing and grounding foods.\n
        •Opt for cooked, moist, and oily foods.\n
        •Prioritize cooked grains like rice and quinoa, well-cooked vegetables, and hearty soups.\n
        •Include ample healthy fats from avocados, nuts, seeds, and ghee.\n
        •Use warming spices like ginger, cinnamon, and cumin.\n
        •Stay hydrated with warm, non-caffeinated herbal teas.'''
    }
    return nutrition_advice.get(vata_category, "")

def get_kapha_nutrition_advice(kapha_category):
    nutrition_advice = {
        "No to Light Kapha": '''\n
        •Warm and light soups with a variety of vegetables.\n
        •Fresh fruits like apples, pears, berries, and pomegranates.\n
        •Whole grains like quinoa, barley, and millet.\n
        •Legumes such as lentils and mung beans.\n
        •Lean proteins like fish and chicken (in moderation).\n
        •Warm herbal teas and spices like ginger, black pepper, and turmeric.''',
        "Moderate Kapha": '''\n
        •Add more pungent spices like cayenne pepper and mustard seeds to increase metabolism.\n
        •Limit dairy products and opt for low-fat or plant-based alternatives.\n
        •Reduce the intake of sweet and heavy fruits like bananas and avocados.\n
        •Warm and dry foods become more important at this stage.\n
        •Avoid cold and heavy foods like ice cream and deep-fried items.\n
        •Include bitter greens like kale, arugula, and dandelion leaves.\n
        •Choose lighter proteins like tofu, tempeh, and lean turkey.\n
        •Use warming spices generously, such as cinnamon, cloves, and cardamom. ''',
        "Extreme Kapha": '''\n
        •Stick to a strict Kapha-pacifying diet with mainly warm, light, and dry foods.\n
        •Focus on steamed or lightly cooked vegetables like asparagus, broccoli, and cauliflower.\n
        •Incorporate more legumes and reduce meat consumption. Avoid sweeteners and processed foods completely.\n
        •Use spices like cayenne, garlic, and ginger to stimulate digestion.'''
    }
    return nutrition_advice.get(kapha_category, "")


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

def predict_kapha_score(input_values):
    kapha_feature_columns = ['MeanBMI', 'SedentaryMinutes', 'LightlyActiveMinutes', 'FairlyActiveMinutes', 'VeryActiveMinutes']

    kapha_model = RandomForestRegressor(n_estimators=100, random_state=42)

    kapha_dataset = pd.read_csv('Kapha_Dataset.csv')
    X_kapha = kapha_dataset[kapha_feature_columns]
    y_kapha = kapha_dataset['Kapha_Score']

    X_kapha_train, _, y_kapha_train, _ = train_test_split(X_kapha, y_kapha, test_size=0.2, random_state=42)

    kapha_model.fit(X_kapha_train, y_kapha_train)

    input_values_df = pd.DataFrame([input_values])

    predicted_kapha_score = kapha_model.predict(input_values_df)
    kapha_category = "No to Light Kapha" if predicted_kapha_score[0] <= 4 else ("Moderate Kapha" if predicted_kapha_score[0] <= 7 else "Extreme Kapha")

    return predicted_kapha_score[0], kapha_category


# Streamlit app
st.title("Dosha Score Prediction")
st.sidebar.title("Enter Dosha Features")

# Dosha selection dropdown
dosha_type = st.sidebar.selectbox("Select Dosha Type", ["Pitta", "Vata", "Kapha"])

if dosha_type == "Pitta":
    dosha_input_values = {
        'AverageHeartRate': st.sidebar.number_input("Average Heart Rate (bpm) (Pitta)"),
        'CumulativeSteps': st.sidebar.number_input("Cumulative Steps (Pitta)"),
        'ActiveDistance': st.sidebar.number_input("Active Distance (km) (Pitta)"),
        'LightActiveDistance': st.sidebar.number_input("Light Active Distance (km) (Pitta)"),
        'MinutesAsleep': st.sidebar.number_input("Minutes Asleep (min) (Pitta)"),
        'Calories': st.sidebar.number_input("Calories (cal) (Pitta)")
    }
    predicted_score, dosha_category, nutrition_advice = predict_pitta_score(dosha_input_values)
    dosha_nutrition_advice = get_pitta_nutrition_advice(dosha_category)
elif dosha_type == "Vata":
    dosha_input_values = {
        'TotalMinutesAsleep': st.sidebar.number_input("Total Minutes Asleep (min) (Vata)"),
        'BedtimeRoutine': st.sidebar.number_input("Bedtime Routine (min) (Vata)"),
        'SleepQuality': st.sidebar.number_input("Sleep Quality (BedtimeRoutine/TotalMinutesAsleep) (Vata)"),
        'TotalSteps': st.sidebar.number_input("Total Steps (Vata)"),
        'SedentaryMinutes': st.sidebar.number_input("Sedentary Minutes (min) (Vata)"),
        'ModeratelyActiveMinutes': st.sidebar.number_input("Moderately Active Minutes (min) (Vata)")
    }
    predicted_score, dosha_category = predict_vata_score(dosha_input_values)
    dosha_nutrition_advice = get_vata_nutrition_advice(dosha_category)
else:
    dosha_input_values = {
        'MeanBMI': st.sidebar.number_input("Mean BMI (Kapha)"),
        'SedentaryMinutes': st.sidebar.number_input("Sedentary Minutes (min) (Kapha)"),
        'LightlyActiveMinutes': st.sidebar.number_input("Lightly Active Minutes (min) (Kapha)"),
        'FairlyActiveMinutes': st.sidebar.number_input("Fairly Active Minutes (min) (Kapha)"),
        'VeryActiveMinutes': st.sidebar.number_input("Very Active Minutes (min) (Kapha)")
    }
    predicted_score, dosha_category = predict_kapha_score(dosha_input_values)
    dosha_nutrition_advice = get_kapha_nutrition_advice(dosha_category)

# Display results for selected dosha
st.write(f"## {dosha_type} Dosha")
st.write(f"Predicted {dosha_type} Score:", predicted_score)
st.write(f"Predicted {dosha_type} Category:", dosha_category)
st.write(f"Nutrition Advice for {dosha_type}:", dosha_nutrition_advice)

import streamlit as st

# ... (previous Streamlit app code)

st.title("Dosha Predominance Prediction")
st.sidebar.title("Enter Heartbeat Value")

# Heartbeat input field
heartbeat_value = st.sidebar.number_input("Enter Heartbeat Value")

if st.sidebar.button("Predict Dosha Predominance"):
    # Call the prediction function with the entered heartbeat value
    predicted_dosha = predict_dosha_predominance(heartbeat_value)

    # Display the predicted Dosha Predominance
    st.write(f"Predicted Dosha Predominance: {predicted_dosha}")


