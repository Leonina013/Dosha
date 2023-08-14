import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

file_path = '/content/drive/My Drive/av/Vata_Dataset_with_Scores.csv'
vata_data = pd.read_csv(file_path)

feature_columns = ['TotalMinutesAsleep', 'BedtimeRoutine', 'SleepQuality', 'TotalSteps', 'SedentaryMinutes', 'ModeratelyActiveMinutes']

X = vata_data[feature_columns]
y = vata_data['Vata_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0]
}

model_xgb = xgb.XGBRegressor(random_state=42)

grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

best_model_xgb = grid_search.best_estimator_

predicted_scores_xgb_test = best_model_xgb.predict(X_test)

rmse_xgb_test = np.sqrt(mean_squared_error(y_test, predicted_scores_xgb_test))
r2_xgb_test = r2_score(y_test, predicted_scores_xgb_test)

print("Root Mean Squared Error (RMSE) on Test Set:", rmse_xgb_test)
print("R-squared (R2) score on Test Set:", r2_xgb_test)

print("Please enter the following information for prediction:")
new_input_values = {
    'TotalMinutesAsleep': int(input("Enter Total Minutes Asleep: ")),
    'BedtimeRoutine': int(input("Enter Bedtime Routine (minutes): ")),
    'SleepQuality': float(input("Enter Sleep Quality (Sleep Quality = BedTimeRoutine/TotalMinutesAsleep): ")),
    'TotalSteps': int(input("Enter Total Steps: ")),
    'SedentaryMinutes': int(input("Enter Sedentary Minutes: ")),
    'ModeratelyActiveMinutes': int(input("Enter Moderately Active Minutes: "))
}

new_input_df = pd.DataFrame([new_input_values])

predicted_vata_score = best_model_xgb.predict(new_input_df)
print("Predicted Vata Score:", predicted_vata_score[0])

category = "No to Light Vata" if predicted_vata_score[0] <= 5 else ("Moderate Vata" if predicted_vata_score[0] <= 8 else "Extreme Vata")

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

print("Category of Vata Dosha:", category)
print("Nutrition Recommendation:", nutrition_advice[category])
