import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

df = pd.read_csv("Machine Learning/realtor-data.csv")

# Drop unnecessary columns
df = df[['bed', 'bath', 'price','house_size','acre_lot']]

# Handle missing values if necessary
df['bed'].fillna(df['bed'].mode()[0], inplace=True)
df['bath'].fillna(df['bath'].mode()[0], inplace=True)

# Drop rows with missing values in the target variable 'price'
df = df.dropna(subset=['price'])

# Train-test split
features = ['bed', 'bath']
target = 'price'
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Save the model and preprocessor
joblib.dump(model, 'linear_regression_model.joblib')

