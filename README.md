import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
try:
    df = pd.read_csv('Ecommerce_Consumer_Behavior_Analysis_Data.csv')
    print("Dataset 'Ecommerce_Consumer_Behavior_Analysis_Data.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'Ecommerce_Consumer_Behavior_Analysis_Data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# --- 1. Data Cleaning and Preprocessing ---

# Clean 'Purchase_Amount': Remove '$' and convert to numeric
df['Purchase_Amount'] = df['Purchase_Amount'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
print("\n'Purchase_Amount' cleaned and converted to numeric.")

# Convert 'Time_of_Purchase' to datetime
# Infer format to handle potential variations, or specify if known (e.g., '%m/%d/%Y')
df['Time_of_Purchase'] = pd.to_datetime(df['Time_of_Purchase'], infer_datetime_format=True, errors='coerce')
# Drop rows where 'Time_of_Purchase' could not be parsed
df.dropna(subset=['Time_of_Purchase'], inplace=True)
print("\n'Time_of_Purchase' converted to datetime and rows with invalid dates dropped.")

# Verify 'Return_Rate' column: Check unique values
unique_return_rates = df['Return_Rate'].unique()
print(f"\nUnique values in 'Return_Rate': {unique_return_rates}")
# Assuming Return_Rate > 0 means returned, and 0 means not returned.
# The original data has 0, 1, 2. If 1 and 2 both mean "returned" in different ways,
# then `> 0` is a good way to binarize.
if not all(val in [0, 1] for val in unique_return_rates):
    print("Warning: 'Return_Rate' column contains values other than 0 or 1. Assuming 1 indicates return.")
    df['IsReturned'] = (df['Return_Rate'] > 0).astype(int)
    print("'Return_Rate' binarized into 'IsReturned' (1 if >0, else 0).")
else:
    df['IsReturned'] = df['Return_Rate']
    print("'Return_Rate' is already binary (0/1) and copied to 'IsReturned'.")


# Create Year and Month columns for potential analysis
df['Purchase_Year'] = df['Time_of_Purchase'].dt.year
df['Purchase_Month'] = df['Time_of_Purchase'].dt.month

print("\nUpdated DataFrame Info after cleaning:")
print(df.info())

# --- 2. Analyze Return Rates ---

print("\n--- Return Rate Analysis ---")

# Overall Return Rate
overall_return_rate = df['IsReturned'].mean() * 100
print(f"Overall Return Rate: {overall_return_rate:.2f}%")

# Return % per Category (using 'Purchase_Category')
return_by_category = df.groupby('Purchase_Category')['IsReturned'].mean().reset_index()
return_by_category['ReturnRate'] = return_by_category['IsReturned'] * 100
return_by_category = return_by_category.sort_values(by='ReturnRate', ascending=False)
print("\nReturn Rate by Product Category:")
print(return_by_category)

# Return % per Location (Geography)
return_by_location = df.groupby('Location')['IsReturned'].mean().reset_index()
return_by_location['ReturnRate'] = return_by_location['IsReturned'] * 100
return_by_location = return_by_location.sort_values(by='ReturnRate', ascending=False)
print("\nReturn Rate by Location (Geography):")
print(return_by_location.head(10)) # Print top 10 locations

# Return % per Purchase Channel (as Marketing Channel proxy)
return_by_channel = df.groupby('Purchase_Channel')['IsReturned'].mean().reset_index()
return_by_channel['ReturnRate'] = return_by_channel['IsReturned'] * 100
return_by_channel = return_by_channel.sort_values(by='ReturnRate', ascending=False)
print("\nReturn Rate by Purchase Channel (Marketing Channel Proxy):")
print(return_by_channel)

# Return % per Brand_Loyalty (as a proxy for 'Supplier' impact)
# Assuming Brand_Loyalty could indicate supplier quality/customer satisfaction with a brand
return_by_brand_loyalty = df.groupby('Brand_Loyalty')['IsReturned'].mean().reset_index()
return_by_brand_loyalty['ReturnRate'] = return_by_brand_loyalty['IsReturned'] * 100
return_by_brand_loyalty = return_by_brand_loyalty.sort_values(by='ReturnRate', ascending=False)
print("\nReturn Rate by Brand Loyalty (Supplier proxy):")
print(return_by_brand_loyalty)

# --- 3. Prepare Data for Logistic Regression ---

# Select features for the model. Choosing a mix of numerical and categorical.
# Selected features based on potential relevance to consumer behavior and returns.
features = [
    'Age', 'Gender', 'Income_Level', 'Marital_Status', 'Education_Level',
    'Occupation', 'Location', 'Purchase_Category', 'Purchase_Amount',
    'Frequency_of_Purchase', 'Purchase_Channel', 'Brand_Loyalty', 'Product_Rating',
    'Time_Spent_on_Product_Research(hours)', 'Social_Media_Influence',
    'Discount_Sensitivity', 'Customer_Satisfaction', 'Engagement_with_Ads',
    'Device_Used_for_Shopping', 'Payment_Method', 'Discount_Used',
    'Customer_Loyalty_Program_Member', 'Purchase_Intent', 'Shipping_Preference',
    'Time_to_Decision'
]
target = 'IsReturned'

# Create a copy to avoid SettingWithCopyWarning
df_model = df.copy()

X = df_model[features]
y = df_model[target]

# Identify categorical columns for one-hot encoding
# Include 'object' (strings) and 'bool' (booleans) as categorical
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

# Convert boolean columns to int (0 or 1) before one-hot encoding if get_dummies doesn't handle them directly
for col in X[categorical_cols].select_dtypes(include=['bool']).columns:
    X[col] = X[col].astype(int)

# Perform one-hot encoding on identified categorical columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True) # drop_first avoids multicollinearity
print("\nFeatures after One-Hot Encoding:")
print(X.head())
print(f"Shape of features (X): {X.shape}")

# Split data into training and testing sets
# stratify=y ensures that the proportion of target variable (IsReturned) is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nTraining data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# --- 4. Build and Train Logistic Regression Model ---

print("\nTraining Logistic Regression Model...")
# 'liblinear' solver is good for smaller datasets and handles L1/L2 regularization.
# max_iter increased to ensure convergence for complex datasets.
model = LogisticRegression(solver='liblinear', random_state=42, max_iter=200)
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model
y_pred = model.predict(X_test)
# predict_proba returns probabilities for both classes (0 and 1). We want probability of class 1 (returned).
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Model Evaluation ---")
print("Classification Report:")
# The classification report shows precision, recall, f1-score, and support for each class.
print(classification_report(y_test, y_pred))
# ROC AUC score measures the area under the Receiver Operating Characteristic curve.
# A score of 0.5 indicates a model no better than random, 1.0 is perfect.
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.2f}")

# Plot ROC curve (optional, for visualization of model performance)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--') # Dashed diagonal line for random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- 5. Predict Probability of Return for all orders ---

print("\nPredicting return probabilities for all orders...")

# Re-apply one-hot encoding on the full DataFrame 'df_model' to ensure consistency
# This is critical to ensure the same columns and order are used for prediction as during training
X_all_processed = pd.get_dummies(df_model[features], columns=categorical_cols, drop_first=True)

# Align columns: This step handles cases where some categories might be present in the full dataset
# but not in the training set (or vice-versa), ensuring the DataFrame for prediction has the
# exact same columns in the exact same order as X_train.
missing_in_all = set(X_train.columns) - set(X_all_processed.columns)
for c in missing_in_all:
    X_all_processed[c] = 0 # Add missing columns to X_all_processed with default value 0

extra_in_all = set(X_all_processed.columns) - set(X_train.columns)
if extra_in_all: # Only drop if there are extra columns
    X_all_processed.drop(columns=list(extra_in_all), inplace=True)

X_all_processed = X_all_processed[X_train.columns] # Ensure order of columns is the same

# Predict probabilities for the entire dataset
df_model['ReturnProbability'] = model.predict_proba(X_all_processed)[:, 1]
print("Return probabilities calculated.")

# --- 6. Identify High-Risk Products/Orders ---

# Define a threshold for high-risk. A common approach is a fixed probability (e.g., > 0.5)
# or taking the top N% of orders by probability.
# Given the ROC AUC of 0.50, a fixed threshold of 0.5 might not yield many 'high-risk' items
# if the model isn't strongly distinguishing. So, we add a fallback to top N%.
high_risk_products = df_model[df_model['ReturnProbability'] > 0.5].sort_values(by='ReturnProbability', ascending=False)

if high_risk_products.empty:
    print("\nNo high-risk products found with a probability threshold > 0.5. Adjusting threshold to top 5% of all orders.")
    num_high_risk = int(len(df_model) * 0.05) # Take top 5% of orders by probability
    high_risk_products = df_model.sort_values(by='ReturnProbability', ascending=False).head(num_high_risk)
    print(f"Identified {len(high_risk_products)} high-risk products (top {num_high_risk} by probability).")
else:
    print(f"\nIdentified {len(high_risk_products)} high-risk products (probability > 0.5).")

print("\nHigh-Risk Products Sample (first 10):")
# Display relevant columns for high-risk products
print(high_risk_products[['Customer_ID', 'Purchase_Category', 'Location', 'Purchase_Amount', 'ReturnProbability', 'IsReturned', 'Return_Rate']].head(10))

# --- 7. Generate CSV of High-Risk Products ---

output_csv_path = 'high_risk_products_ecommerce.csv'
# Select columns relevant for the high-risk products CSV
high_risk_products[['Customer_ID', 'Time_of_Purchase', 'Purchase_Category', 'Location', 'Purchase_Channel', 'Purchase_Amount', 'ReturnProbability', 'IsReturned', 'Return_Rate']].to_csv(output_csv_path, index=False)
print(f"\nHigh-risk products saved to: {output_csv_path}")

# --- Optional: Save the full processed dataframe for Power BI if needed ---
# This file will contain all original columns plus 'IsReturned' and 'ReturnProbability'.
# It's useful for Power BI to have a single, comprehensive data source.
full_processed_data_path = 'ecommerce_processed_data_for_powerbi.csv'
df_model.to_csv(full_processed_data_path, index=False)
print(f"\nFull processed data (including IsReturned and ReturnProbability) saved to: {full_processed_data_path}")

print("\n--- Python script execution complete. ---")
print("You now have 'high_risk_products_ecommerce.csv' and 'ecommerce_processed_data_for_powerbi.csv' for further analysis and dashboarding.")
