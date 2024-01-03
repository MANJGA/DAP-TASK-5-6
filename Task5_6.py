import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

# df = pd.read_csv('example.csv') CSV-ის გარეშე მაქვს ზეპირად დაწერილი კოდი.

# Task 1: Simple Linear Regression
X = df[['feature_column']].values
y = df['target_column'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
y_pred_simple = model_simple.predict(X_test)
r_squared_simple = r2_score(y_test, y_pred_simple)

# Task 2: Multiple Linear Regression
X_multi = df[['feature1', 'feature2', 'feature3']].values
X_train_multi, X_test_multi, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=0)
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train)
y_pred_multi = model_multi.predict(X_test_multi)
r_squared_multi = r2_score(y_test, y_pred_multi)

# Task 3: Decision Tree Regression
model_tree_reg = DecisionTreeRegressor(random_state=0)
model_tree_reg.fit(X_train, y_train)
y_pred_tree_reg = model_tree_reg.predict(X_test)
r_squared_tree_reg = r2_score(y_test, y_pred_tree_reg)

# Task 4: Logistic Regression
y_binary = df['binary_target_column'].values
X_train_log, X_test_log, y_train_binary, y_test_binary = train_test_split(X_multi, y_binary, test_size=0.2, random_state=0)
model_logistic = LogisticRegression()
model_logistic.fit(X_train_log, y_train_binary)
y_pred_logistic = model_logistic.predict(X_test_log)
accuracy_logistic = accuracy_score(y_test_binary, y_pred_logistic)

# Task 5: Decision Tree Classification
model_tree_class = DecisionTreeClassifier(random_state=0)
model_tree_class.fit(X_train_log, y_train_binary)
y_pred_tree_class = model_tree_class.predict(X_test_log)
accuracy_tree_class = accuracy_score(y_test_binary, y_pred_tree_class)

# Print out the efficiency metrics
print(f"Simple Linear Regression R-squared: {r_squared_simple:.2f}")
print(f"Multiple Linear Regression R-squared: {r_squared_multi:.2f}")
print(f"Decision Tree Regression R-squared: {r_squared_tree_reg:.2f}")
print(f"Logistic Regression Accuracy: {accuracy_logistic:.2f}")
print(f"Decision Tree Classification Accuracy: {accuracy_tree_class:.2f}")


