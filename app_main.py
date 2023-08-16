import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  
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
        "No to Light Pitha": '''
        Favor cooling foods: Incorporate foods that have a cooling effect on the body.
        Include sweet, bitter, and astringent tastes: Focus on foods with these tastes to balance excess heat.
        Limit spicy, oily, and acidic foods: Reduce or avoid foods that can increase Pitta, such as hot spices, fried foods, and excessive sourness.
        Stay hydrated: Drink cool or room temperature water and herbal teas to balance the heat.
        Enjoy fresh, ripe, and sweet fruits: Opt for sweet fruits like melons, grapes, and sweet berries.
        Sample No to Light Pitta Diet:
        - Breakfast: A bowl of cool oatmeal with ripe berries.
        - Lunch: Steamed vegetables with quinoa and a cooling mint-cucumber yogurt sauce.
        - Snack: Sliced watermelon or a cool cucumber salad.
        - Dinner: Basmati rice with steamed zucchini and a small serving of sweet and juicy fruits.
        ''',
        "Moderate Pitha": '''
        Balanced meals: Include a mix of cooling and slightly warming foods to maintain equilibrium.
        Enjoy a variety of tastes: Incorporate sweet, bitter, astringent, and a moderate amount of pungent tastes.
        Moderation in spices: Use milder spices like coriander, fennel, and cardamom in your meals.
        Stay hydrated: Drink room temperature water and herbal teas.
        Include fresh and cooked foods: Combine raw and cooked vegetables, grains, and legumes.
        Sample Moderate Pitta Diet:
        - Breakfast: A fruit smoothie with ripe bananas, mangoes, and a pinch of cardamom.
        - Lunch: Mixed greens salad with roasted vegetables, quinoa, and a lemon-olive oil dressing.
        - Snack: Sliced apples with almond butter.
        - Dinner: Baked salmon with steamed asparagus and a side of basmati rice.
        ''',
        "Extreme Pitha": '''
        Emphasize cooling and calming foods: Prioritize foods with a strong cooling effect.
        Favor sweet, bitter, and astringent tastes: These tastes help balance excess heat and acidity.
        Avoid hot and spicy foods: Steer clear of very spicy, oily, and fried foods.
        Stay well-hydrated: Drink cool water, coconut water, and herbal teas.
        Include plenty of fresh fruits: Opt for sweet, juicy fruits that help cool the body.
        Sample High Pitta Diet:
        - Breakfast: A bowl of cool, cooked barley cereal with sliced peaches.
        - Lunch: Cucumber and mint raita with rice or quinoa and a side of steamed broccoli.
        - Snack: A handful of sweet grapes.
        - Dinner: Baked or steamed white fish with a side of lightly steamed carrots and zucchini.
        '''
    }
    return nutrition_advice.get(pitta_category, "")


def get_vata_nutrition_advice(vata_category):
    nutrition_advice = {
        "No to Light Vata": '''Favor warm, cooked, and easily digestible foods.
Incorporate plenty of healthy fats such as ghee, coconut oil, and olive oil.
Include nourishing and grounding foods like sweet potatoes, whole grains (cooked), cooked vegetables, and lentils.
Drink warm herbal teas (non-caffeinated) like ginger, cinnamon, and licorice.
Reduce raw foods, cold foods, and excessive caffeine.''',
        "Moderate Vata": '''Favor warm, cooked, and easily digestible foods.
Incorporate plenty of healthy fats such as ghee, coconut oil, and olive oil.
Include nourishing and grounding foods like sweet potatoes, whole grains (cooked), cooked vegetables, and lentils.
Drink warm herbal teas (non-caffeinated) like ginger, cinnamon, and licorice.
Reduce raw foods, cold foods, and excessive caffeine.
Include a variety of cooked vegetables, grains, and legumes.
Incorporate small amounts of dairy, if tolerated (e.g., warm milk with spices).
Include foods with mild natural sweetness like ripe fruits (in moderation) and sweet spices.
Hydrate well with warm water, herbal teas, and warm soups.''',
        "Extreme Vata": '''Focus on stabilizing and grounding foods.
Opt for cooked, moist, and oily foods.
Prioritize cooked grains like rice and quinoa, well-cooked vegetables, and hearty soups.
Include ample healthy fats from avocados, nuts, seeds, and ghee.
Use warming spices like ginger, cinnamon, and cumin.
Stay hydrated with warm, non-caffeinated herbal teas.'''
    }
    return nutrition_advice.get(vata_category, "")

def get_kapha_nutrition_advice(kapha_category):
    nutrition_advice = {
        "No to Light Kapha": '''Warm and light soups with a variety of vegetables. Fresh fruits like apples, pears, berries, and pomegranates. Whole grains like quinoa, barley, and millet. Legumes such as lentils and mung beans. Lean proteins like fish and chicken (in moderation). Warm herbal teas and spices like ginger, black pepper, and turmeric.''',
        "Moderate Kapha": '''Add more pungent spices like cayenne pepper and mustard seeds to increase metabolism. Limit dairy products and opt for low-fat or plant-based alternatives. Reduce the intake of sweet and heavy fruits like bananas and avocados. Warm and dry foods become more important at this stage. Avoid cold and heavy foods like ice cream and deep-fried items. Include bitter greens like kale, arugula, and dandelion leaves. Choose lighter proteins like tofu, tempeh, and lean turkey. Use warming spices generously, such as cinnamon, cloves, and cardamom. ''',
        "Extreme Kapha": '''Stick to a strict Kapha-pacifying diet with mainly warm, light, and dry foods. Focus on steamed or lightly cooked vegetables like asparagus, broccoli, and cauliflower. Incorporate more legumes and reduce meat consumption. Avoid sweeteners and processed foods completely. Use spices like cayenne, garlic, and ginger to stimulate digestion.'''
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

# Sidebar inputs for Pitta Dosha
pitta_input_values = {
    'AverageHeartRate': st.sidebar.number_input("Average Heart Rate"),
    'CumulativeSteps': st.sidebar.number_input("Cumulative Steps"),
    'ActiveDistance': st.sidebar.number_input("Active Distance"),
    'LightActiveDistance': st.sidebar.number_input("Light Active Distance"),
    'MinutesAsleep': st.sidebar.number_input("Minutes Asleep"),
    'Calories': st.sidebar.number_input("Calories")
}

# Sidebar inputs for Vata Dosha
vata_input_values = {
    'TotalMinutesAsleep': st.sidebar.number_input("Total Minutes Asleep (Vata)"),
    'BedtimeRoutine': st.sidebar.number_input("Bedtime Routine (Vata)"),
    'SleepQuality': st.sidebar.number_input("Sleep Quality (Vata)"),
    'TotalSteps': st.sidebar.number_input("Total Steps (Vata)"),
    'SedentaryMinutes': st.sidebar.number_input("Sedentary Minutes (Vata)"),
    'ModeratelyActiveMinutes': st.sidebar.number_input("Moderately Active Minutes (Vata)")
}

# Sidebar inputs for Kapha Dosha
kapha_input_values = {
    'MeanBMI': st.sidebar.number_input("Mean BMI"),
    'SedentaryMinutes': st.sidebar.number_input("Sedentary Minutes"),
    'LightlyActiveMinutes': st.sidebar.number_input("Lightly Active Minutes"),
    'FairlyActiveMinutes': st.sidebar.number_input("Fairly Active Minutes"),
    'VeryActiveMinutes': st.sidebar.number_input("Very Active Minutes")
}

# Predict scores for Pitta, Vata, and Kapha
predicted_pitta_score, pitta_category, pitta_nutrition_advice = predict_pitta_score(pitta_input_values)
predicted_vata_score, vata_category = predict_vata_score(vata_input_values)
predicted_kapha_score, kapha_category = predict_kapha_score(kapha_input_values)

# Display results for Pitta, Vata, and Kapha
st.write("## Pitta Dosha")
st.write("Predicted Pitta Score:", predicted_pitta_score)
st.write("Predicted Pitta Category:", pitta_category)
st.write("Nutrition Advice for Pitta:", pitta_nutrition_advice)

st.write("## Vata Dosha")
st.write("Predicted Vata Score:", predicted_vata_score)
st.write("Predicted Vata Category:", vata_category)
vata_nutrition_advice = get_vata_nutrition_advice(vata_category)
st.write("Nutrition Advice for Vata:", vata_nutrition_advice)


st.write("## Kapha Dosha")
st.write("Predicted Kapha Score:", predicted_kapha_score)
st.write("Predicted Kapha Category:", kapha_category)
kapha_nutrition_advice = get_kapha_nutrition_advice(kapha_category)
st.write("Nutrition Advice for Kapha:", kapha_nutrition_advice)
