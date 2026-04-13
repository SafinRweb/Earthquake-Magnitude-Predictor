import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load Data
df = pd.read_csv('BD-IND-NPL-BHUTN-MYNMR.csv')
data = df.dropna(subset=['latitude', 'longitude', 'depth', 'mag'])

X = data[['latitude', 'longitude', 'depth']]
y = data['mag']

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train THREE Models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# 4. Compare Results
print("=== Model Comparison (Mean Absolute Error) ===")
print(f"Linear Regression MAE:     {mean_absolute_error(y_test, lr_pred):.4f}")
print(f"Random Forest MAE:         {mean_absolute_error(y_test, rf_pred):.4f}")
print(f"Gradient Boosting MAE:     {mean_absolute_error(y_test, gb_pred):.4f}")

# 5. Generate & Save 6 Graphs

# --- Group 1: Actual vs Predicted Graphs ---
# 1. Linear Regression
plt.figure(figsize=(8, 5))
plt.scatter(y_test, lr_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Magnitudes (Linear Regression)')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.grid(True)
plt.savefig('actual_vs_predicted_lr.png')
plt.close()

# 2. Random Forest
plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_pred, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Magnitudes (Random Forest)')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.grid(True)
plt.savefig('actual_vs_predicted_rf.png')
plt.close()

# 3. Gradient Boosting
plt.figure(figsize=(8, 5))
plt.scatter(y_test, gb_pred, alpha=0.5, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Magnitudes (Gradient Boosting)')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.grid(True)
plt.savefig('actual_vs_predicted_gb.png')
plt.close()

# --- Group 2: Residual Error Plots ---
# 4. Linear Regression Residuals
plt.figure(figsize=(8, 5))
plt.scatter(lr_pred, y_test - lr_pred, alpha=0.5, color='purple')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.title('Residual Plot (Linear Regression)')
plt.xlabel('Predicted Magnitude')
plt.ylabel('Residual Error (Actual - Predicted)')
plt.grid(True)
plt.savefig('residual_error_plot_lr.png')
plt.close()

# 5. Random Forest Residuals
plt.figure(figsize=(8, 5))
plt.scatter(rf_pred, y_test - rf_pred, alpha=0.5, color='teal')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.title('Residual Plot (Random Forest)')
plt.xlabel('Predicted Magnitude')
plt.ylabel('Residual Error (Actual - Predicted)')
plt.grid(True)
plt.savefig('residual_error_plot_rf.png')
plt.close()

# 6. Gradient Boosting Residuals
plt.figure(figsize=(8, 5))
plt.scatter(gb_pred, y_test - gb_pred, alpha=0.5, color='brown')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.title('Residual Plot (Gradient Boosting)')
plt.xlabel('Predicted Magnitude')
plt.ylabel('Residual Error (Actual - Predicted)')
plt.grid(True)
plt.savefig('residual_error_plot_gb.png')
plt.close()
