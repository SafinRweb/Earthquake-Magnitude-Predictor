import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load Data
df = pd.read_csv('BD-IND-NPL-BHUTN-MYNMR.csv')
data = df.dropna(subset=['latitude', 'longitude', 'depth', 'mag'])

X = data[['latitude', 'longitude', 'depth']]
y = data['mag']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train THREE Models
# Model 1: Linear Regression (Baseline)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)

# Model 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)

# Model 3: Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_pred)

# Print MAE scores to terminal
print("=== Model Comparison (Mean Absolute Error) ===")
print(f"Linear Regression MAE:     {lr_mae:.4f}")
print(f"Gradient Boosting MAE:     {gb_mae:.4f}")
print(f"Random Forest MAE:         {rf_mae:.4f}")
print("-" * 45)


# Graph 1: Correlation Matrix Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = data[['latitude', 'longitude', 'depth', 'mag']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f", 
            linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix of Seismic Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Feature Importance (Random Forest)
plt.figure(figsize=(8, 5))
features = ['Latitude', 'Longitude', 'Depth']
importances = rf_model.feature_importances_
colors = ['#4C72B0', '#55A868', '#C44E52']
plt.bar(features, importances, color=colors)
plt.title('Feature Importance Analysis (Random Forest)')
plt.ylabel('Importance Score')
for i, v in enumerate(importances):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
plt.savefig('feature_importance.png')
plt.close()

# Model Performance Comparison Bar Chart
plt.figure(figsize=(8, 5))
models = ['Linear Regression', 'Gradient Boosting', 'Random Forest']
maes = [lr_mae, gb_mae, rf_mae]
colors_mae = ['#2ECC71', '#F39C12', '#E74C3C'] # Green (best) to Red (worst)

bars = plt.bar(models, maes, color=colors_mae)
plt.title('Algorithm Comparison: Mean Absolute Error (Lower is Better)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.ylim(0.25, 0.35) # Zoom in to show the difference clearly
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f"{yval:.4f}", ha='center', fontweight='bold')
plt.savefig('model_comparison_bar.png')
plt.close()

# Actual vs Predicted (Linear Regression)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, lr_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Magnitudes (Linear Regression)')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.grid(True)
plt.savefig('actual_vs_predicted_lr.png')
plt.close()

# Residual Error Plot (Linear Regression)
lr_residuals = y_test - lr_pred
plt.figure(figsize=(8, 5))
plt.scatter(lr_pred, lr_residuals, alpha=0.5, color='purple')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.title('Residual Error Plot (Linear Regression)')
plt.xlabel('Predicted Magnitude')
plt.ylabel('Residual Error (Actual - Predicted)')
plt.grid(True)
plt.savefig('residual_error_plot_lr.png')
plt.close()