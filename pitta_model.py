import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


dataset_file_path = 'Pitha_Dataset.csv'
dataset = pd.read_csv(dataset_file_path)


input_feature_columns = ['AverageHeartRate', 'CumulativeSteps', 'ActiveDistance', 'LightActiveDistance', 'MinutesAsleep', 'Calories']


X = dataset[input_feature_columns]
y = dataset['Pitha_Score']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


model.fit(X_train_scaled, y_train)


print("Please enter the following information for prediction:")
new_input_values = {}
for column in input_feature_columns:
    new_input_values[column] = float(input(f"Enter {column}: "))
new_input_df = pd.DataFrame([new_input_values])
new_input_df_scaled = scaler.transform(new_input_df)


predicted_pitha_score = model.predict(new_input_df_scaled)


def get_pitha_category(pitha_score):
    if pitha_score <= 5:
        return "No to Light Pitha"
    elif pitha_score <= 8:
        return "Moderate Pitha"
    else:
        return "Extreme Pitha"


def get_nutrition_advice(pitha_category):
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
        High Pitta Dosha (>9):
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
    return nutrition_advice[pitha_category]


predicted_pitha_category = get_pitha_category(predicted_pitha_score[0])
print("Predicted Pitha Category:", predicted_pitha_category)
print("Nutrition Advice:")
print(get_nutrition_advice(predicted_pitha_category))
