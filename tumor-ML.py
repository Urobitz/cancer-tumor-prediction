import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

#open file
df = pd.read_csv("breast_cancer_enhanced_dataset.csv")
#X is the inputs we are going to use
x = df[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
        "smoothness_mean", "compactness_mean", "concavity_mean", 
        "concave points_mean", "shape_irregularity", "border_complexity",  "radius_texture_interaction", 
        "radius_concavity_interaction", "concavity_density"]]
#Y is the output or what we want to predict
y = df["diagnosis"]

#divides each variable in 2, thats why there are 4 variables
x_train_input, x_test_input, y_train_input, y_test_input = train_test_split(x,y,test_size=0.2, random_state=42)

#Since is for a % we use a regressor which predicts numbers, in contrast to classifier, which works for category
model = RandomForestClassifier()
#train the model
model.fit(x_train_input, y_train_input)
#predictions
predictions = model.predict(x_test_input)

custom_tumor = pd.DataFrame([[14.5, 19.5, 92.0, 650.0, 0.10, 0.08, 0.05, 0.03, 0.001, 0.02, 257.0, 0.18, 0.00003]], 
                         columns=["radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                                  "smoothness_mean", "compactness_mean", "concavity_mean",
                                  "concave points_mean", "shape_irregularity", "border_complexity", "radius_texture_interaction",
                                  "radius_concavity_interaction", "concavity_density"])
print("Tumor Type: ", model.predict(custom_tumor))

importances = model.feature_importances_
features = x.columns

plt.bar(features, importances)
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()