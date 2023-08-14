import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

dataset_filepath = 'Kapha_Dataset.csv'
df = pd.read_csv(dataset_filepath)

X = df[['MeanBMI', 'SedentaryMinutes', 'LightlyActiveMinutes', 'FairlyActiveMinutes', 'VeryActiveMinutes']]
y = df['Kapha_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Root Mean Squared Error on the test set:", rmse)

def get_nutrition_recommendations(kapha_score):
    if kapha_score >= 0 and kapha_score <= 4:
        return "Perfect Kapha Range. Maintain your diet, here are some suggestions.", "Warm and light soups with a variety of vegetables. Fresh fruits like apples, pears, berries, and pomegranates. Whole grains like quinoa, barley, and millet. Legumes such as lentils and mung beans. Lean proteins like fish and chicken (in moderation). Warm herbal teas and spices like ginger, black pepper, and turmeric."
    elif kapha_score > 4 and kapha_score <= 5:
        return "Mild Kapha Imbalance.", "Add more pungent spices like cayenne pepper and mustard seeds to increase metabolism. Limit dairy products and opt for low-fat or plant-based alternatives. Reduce the intake of sweet and heavy fruits like bananas and avocados."
    elif kapha_score > 5 and kapha_score <= 7:
        return "Moderate Kapha Imbalance.", "Warm and dry foods become more important at this stage. Avoid cold and heavy foods like ice cream and deep-fried items. Include bitter greens like kale, arugula, and dandelion leaves. Choose lighter proteins like tofu, tempeh, and lean turkey. Use warming spices generously, such as cinnamon, cloves, and cardamom."
    elif kapha_score > 7 and kapha_score <= 11:
        return "Extreme Kapha Imbalance.", "Stick to a strict Kapha-pacifying diet with mainly warm, light, and dry foods. Focus on steamed or lightly cooked vegetables like asparagus, broccoli, and cauliflower. Incorporate more legumes and reduce meat consumption. Avoid sweeteners and processed foods completely. Use spices like cayenne, garlic, and ginger to stimulate digestion."
    else:
        return "Invalid Kapha Score. Please provide a valid Kapha Score in the range of 0 to 11.", ""

random_inputs = pd.DataFrame({
    'MeanBMI': [16.5, 18.8, 22.0, 25.0, 28.5],
    'SedentaryMinutes': [1400, 400, 220, 600, 700],
    'LightlyActiveMinutes': [2, 300, 285, 500, 600],
    'FairlyActiveMinutes': [5, 50, 60, 70, 80],
    'VeryActiveMinutes': [10, 70, 30, 90, 100]
})

predicted_kapha_score = rf_regressor.predict(random_inputs)

results_df = pd.DataFrame(random_inputs)
results_df['Predicted Kapha Score'] = predicted_kapha_score
results_df['Nutrition Recommendations'], results_df['Foods'] = zip(*results_df['Predicted Kapha Score'].map(get_nutrition_recommendations))

results_filepath = '/content/drive/My Drive/av/Kapha_Results.csv'
results_df.to_csv(results_filepath, index=False)

print("Results saved to:", results_filepath)
