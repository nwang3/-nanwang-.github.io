import inspect
import pprint
import numpy as np 
import pandas as pd 
import os
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib


df = pd.read_csv("Machine Learning/realtor-data.csv")
df.head()
num_countries = df['state'].unique()
print("Number of different states:", num_countries)
state_stats = df.groupby('state')['price'].agg(['mean', 'median', 'min', 'max'])
print(state_stats)
state_mean_price = df.groupby('state')['price'].mean().sort_values(ascending = False)
plt.figure(figsize=(12, 6))
state_mean_price.plot(kind='bar')
plt.xlabel('State')
plt.ylabel('Mean Price')
plt.title('Mean Price by State')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
df.describe()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
total_missing = df.isna().sum()*100/len(df)
print('Percentage Missing Value %')
total_missing
df['bed'].fillna(df['bed'].mode()[0], inplace=True)
df['bath'].fillna(df['bath'].mode()[0], inplace=True)
df['acre_lot'].fillna(df['acre_lot'].mode()[0], inplace=True)
df['house_size'].fillna(df['house_size'].mode()[0], inplace=True)
df = df.dropna(subset=['zip_code','city'])
df = df.drop('prev_sold_date', axis=1)
df.info()
column_num = ['bed', 'bath', 'acre_lot', 'house_size', 'price']
fig = px.box(df[column_num], labels={'variable': 'Column', 'value': 'Value'}, title='Outlier Before Remove')
fig.update_xaxes(tickvals=[1, 2, 3, 4, 5], ticktext=column_num)
fig.show()
print(f'Total Rows with Outliers: {df.shape[0]}')
Q1 = df[column_num].quantile(0.25)
Q3 = df[column_num].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[column_num] < (Q1 - 1.5 * IQR)) | (df[column_num] > (Q3 + 1.5 * IQR))).any(axis=1)]
print(IQR)
column_num = ['bed', 'bath', 'acre_lot', 'house_size', 'price']
df_selected = df[column_num]
df_selected_melted = df_selected.melt(var_name='Column', value_name='Value')
fig = px.box(df_selected_melted, x='Column', y='Value', title='Outlier After Remove')
fig.update_xaxes(tickvals=[1, 2, 3, 4, 5], ticktext=column_num)
fig.show()
print(f'Total Rows Without Outliers: {df.shape[0]}')
summary_stats = df.describe()
print(summary_stats)

numeric_columns = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
categorical_column = 'state'
value_counts = df[categorical_column].value_counts(ascending = False)
print(value_counts)

df['price_per_sqft'] = df['price'] / df['house_size']
price_per_sqft_by_city = df.groupby('state')['price_per_sqft'].mean()
print(price_per_sqft_by_city)
z_scores = (df['price_per_sqft'] - df['price_per_sqft'].mean()) / df['price_per_sqft'].std()
outlier_threshold = 3
outliers = df[z_scores.abs() > outlier_threshold]
df = df[z_scores.abs() <= outlier_threshold]
avg_price_per_sqft_by_city = df.groupby('state')['price_per_sqft'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_price_per_sqft_by_city.index, y=avg_price_per_sqft_by_city.values)
plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Average Price per Square Foot')
plt.title('Average Price per Square Foot by State')
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(df['house_size'], df['price_per_sqft'])
plt.xlabel('House Size')
plt.ylabel('Price per Square Foot')
plt.title('Price per Square Foot vs. House Size')
plt.show()
df.info()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
features = ['status', 'bed', 'bath', 'acre_lot', 'city', 'state', 'zip_code', 'house_size']
target = 'price'
cat_features = ['status', 'city', 'state', 'zip_code']
num_features = ['bed', 'bath', 'acre_lot', 'house_size']
preprocessor = ColumnTransformer([('encoder', OneHotEncoder(), cat_features)], remainder='passthrough')
encoded_features = preprocessor.fit_transform(df[features])
X_train, X_test, y_train, y_test = train_test_split(encoded_features, df[target], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
print(df.columns)
numerical_features = ['bed', 'bath', 'acre_lot', 'house_size']
target = 'price'
X_train, X_test, y_train, y_test = train_test_split(df[numerical_features], df[target], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
X = df.drop(['price', 'price_per_sqft'], axis=1)
y = df['price']
categorical_cols = ['status', 'city', 'state']
X_encoded = pd.get_dummies(X, columns=categorical_cols)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print(pd.DataFrame(zip(X_encoded.columns, model.coef_)))

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

joblib.dump(preprocessor, 'preprocessor.joblib')

joblib.dump(model, 'linear_regression_model.joblib')

loaded_preprocessor = joblib.load('preprocessor.joblib')