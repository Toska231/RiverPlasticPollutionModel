#!/usr/bin/env python
# coding: utf-8

# # Predictive Modelling and Visual Risk Intelligence Dashboard for River Plastic Waste (2015–2060)
# 
# Dataset Source: https://www.kaggle.com/datasets/khushikyad001/river-plastic-waste-risk-scenarios-2015-vs-2060

# # Columns
# 
# • Country
# • Continent
# • River Name
# • River Length (km)
# • Annual Plastic Waste (Tonnes)
# • Population the near RivUrbanisationzation Rate (%)
# • Waste Management Quality (Score)
# • Plastic Waste Mismanaged (%)
# • Risk Index
# • Projected Plastic Waste (2060)ste (2060)

# # 1. Data Understanding & Cleaning (Python / Pandas)
# • Create consistent time-based records.
# • Handle missing values,normalisee columns, andcategorisee risk levels.
# • Feature engineering: Waste per capita, risk density, etc.

# # Step 1: Data Loading & Initial Inspection

# In[1]:


# Import pandas library first
import pandas as pd

# Now you can use pd to read the CSV file
data = pd.read_csv('1746634473298_d36f75e695.csv')
data.head(10)


# # 2. Basic inspection

# In[2]:


print(data.shape)
print(data.info())
print(data.head())


# # Step 2: Data Cleaning & Feature Engineering

# In[3]:


# Create derived features
data["Plastic_Waste_per_Capita_2015"] = data["Plastic_Waste_2015_tons"] / data["Population_2015"]
data["Plastic_Waste_Density_2015"] = data["Plastic_to_River_2015_tons"] / data["River_Length_km"]

def categorize_risk(score):
    if score >= 100:
        return "High"
    elif score <= -100:
        return "Low"
    else:
        return "Medium"

data["Risk_Level_Change"] = data["Risk_Score_Change"].apply(categorize_risk)

# Drop unused column
data.drop(columns=["River_ID"], inplace=True)


# # 3. Exploratory Data Analysis (EDA)
# 
# • Identify the top 10 highest-risk rivers in 2015 and 2060.
# • Comparethe  continents’ performance (worsening vs improving trends).
# • Visual storytelling using Seaborn/Matplotlib.

# In[4]:


# Uncomment the seaborn import
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # Added pandas import for pd.melt

sns.set(style="whitegrid")

# Top 10 Rivers by Plastic Waste 2015 and 2060
top_2015 = data.sort_values("Plastic_Waste_2015_tons", ascending=False).head(10)
top_2060 = data.sort_values("Plastic_Waste_2060_tons", ascending=False).head(10)

# Plot 2015
plt.figure(figsize=(10, 6))
sns.barplot(x="Plastic_Waste_2015_tons", y="River_Name", data=top_2015, palette="Blues_d")
plt.title("Top 10 Rivers by Plastic Waste in 2015")
plt.tight_layout()
plt.show()

# Plot 2060
plt.figure(figsize=(10, 6))
sns.barplot(x="Plastic_Waste_2060_tons", y="River_Name", data=top_2060, palette="Reds_d")
plt.title("Top 10 Rivers by Projected Plastic Waste in 2060")
plt.tight_layout()
plt.show()

# Continent-wise comparison
continent_summary = data.groupby("Continent")[["Plastic_Waste_2015_tons", "Plastic_Waste_2060_tons"]].sum().reset_index()
continent_melted = pd.melt(continent_summary, id_vars="Continent", var_name="Year", value_name="Plastic_Waste_Tons")
# Correcting the regex pattern to capture 4 digits
continent_melted["Year"] = continent_melted["Year"].str.extract(r"(\d{4})")

plt.figure(figsize=(10, 6))
sns.barplot(data=continent_melted, x="Continent", y="Plastic_Waste_Tons", hue="Year")
plt.title("Plastic Waste by Continent: 2015 vs 2060")
plt.tight_layout()
plt.show()


# # 4. Predictive Modelling (ML & DL)
# 
# • Goal: Predict 2060 Plastic Waste (Tonnes) based on 2015 data.
# • Apply regression models: Linear Regression, Random Forest, XGBoost.
# • Deploy a Deep Learning model (e.g., MLPRegressor using Keras/TensorFlow).
# • Evaluate performance using MAE, RMSE, and R².

# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# Feature selection
features = [
    "Plastic_Waste_2015_tons",
    "Mismanaged_Waste_2015_tons",
    "Plastic_to_River_2015_tons",
    "River_Length_km",
    "Urbanization_2015_pct",
    "GDP_per_capita_2015",
    "Policy_Strength_2015",
    "Waste_Collection_Rate_2015",
    "Plastic_Waste_per_Capita_2015",
    "Plastic_Waste_Density_2015"
]
target = "Plastic_Waste_2060_tons"

X = data[features]
y = data[target]

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0)
}

# Evaluation
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name} Results:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 Score:", r2_score(y_test, y_pred))


# # 4. Risk Classification Model
# 
# • Convert Risk Index into categories (Low, Medium, High).
# • Train classification models (Logistic Regression, SVM, Random Forest).
# • Predict country-wise classification and accuracy.

# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # ✅ SVM import added
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Encode Risk Level
data["Risk_Level_Encoded"] = data["Risk_Level_Change"].map({"Low": 0, "Medium": 1, "High": 2})

X_cls = data[features]
y_cls = data["Risk_Level_Encoded"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
X_train_cls_scaled = scaler.fit_transform(X_train_cls)
X_test_cls_scaled = scaler.transform(X_test_cls)

# Add SVM to classifier models
cls_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "Support Vector Machine (SVM)": SVC()
}

# Train and evaluate all classifiers
for name, model in cls_models.items():
    model.fit(X_train_cls_scaled, y_train_cls)
    y_pred_cls = model.predict(X_test_cls_scaled)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test_cls, y_pred_cls))
    print("Confusion Matrix:\n", confusion_matrix(y_test_cls, y_pred_cls))


# In[8]:


# Classifier Performance Comparison

import matplotlib.pyplot as plt
import numpy as np

# Define model names
models = ['Logistic Regression', 'Random Forest', 'SVM']

# Define metrics from the classification reports
accuracy = [0.63, 0.63, 0.61]
macro_f1 = [0.60, 0.62, 0.60]
weighted_f1 = [0.62, 0.63, 0.61]

# Plot setup
x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for each metric
bars1 = ax.bar(x - width, accuracy, width, label='Accuracy')
bars2 = ax.bar(x, macro_f1, width, label='Macro F1')
bars3 = ax.bar(x + width, weighted_f1, width, label='Weighted F1')

# Add labels and formatting
ax.set_ylabel('Score')
ax.set_title('Classifier Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.5, 0.7)
ax.legend()

# Annotate scores on each bar
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

# Show plot
plt.tight_layout()
plt.show()


# # 5. Trend Forecasting
# 
# • Use Time-Series forecasting if multi-year data is engineered.
# • Prophet or LSTM for multi-decade waste projection (optional but adds depth)

# In[4]:


# Time-Series Forecasting with Prophet

# Import pandas first
import pandas as pd  # Added this import statement for pandas
import matplotlib.pyplot as plt  # Added import for plt

# First, install the prophet package
get_ipython().system('pip install prophet')

# Import after installation
from prophet import Prophet

# Create sample data since 'df' is not defined
# Replace this with your actual data loading code
df = pd.DataFrame({
    'Plastic_Waste_2015_tons': [300000000],  # Example value
    'Plastic_Waste_2060_tons': [1000000000]  # Example value
})

# Example: Global yearly plastic waste trend (2015 & 2060 only)
yearly_data = pd.DataFrame({
    'ds': ["2015-01-01", "2060-01-01"],
    'y': [df["Plastic_Waste_2015_tons"].sum(), df["Plastic_Waste_2060_tons"].sum()]
})

model = Prophet()
model.fit(yearly_data)

# Forecast future to 2080
future = model.make_future_dataframe(periods=20, freq='Y')
forecast = model.predict(future)

# Plot forecast
fig = model.plot(forecast)
plt.title("Global Plastic Waste Forecast (2015–2080)")
plt.show()

