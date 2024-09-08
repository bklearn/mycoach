import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Simulating a dataset
np.random.seed(42)  # For reproducibility

# Create synthetic data
data = pd.DataFrame({
    'Age': np.random.randint(17, 25, 1000),
    'Gender': np.random.choice(['Male', 'Female'], 1000),
    'SocioeconomicStatus': np.random.randint(1, 5, 1000),  # 1 to 4 scale
    'AcademicPerformance': np.random.randint(1, 100, 1000),  # 1 to 100 scale
    'Region': np.random.choice(['Urban', 'Rural', 'Suburban'], 1000),
    'ParentalEducation': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'PredictedChoice': np.random.choice(['University', 'Vocational Training', 'Apprenticeship', 'Direct Employment'], 1000),
    'EmploymentRate': np.random.uniform(50, 100, 1000),  # Mock employment rate
    'ExpectedSalary': np.random.randint(30000, 80000, 1000)  # Mock expected salary
})

# Convert categorical features to numerical encoding
data = pd.get_dummies(data, columns=['Gender', 'Region', 'ParentalEducation'], drop_first=True)
data.head()

### Model Building for Predicting Education and Training Choices


# Features and target variable for predicting educational choices
X = data.drop(['PredictedChoice', 'EmploymentRate', 'ExpectedSalary'], axis=1)
y = data['PredictedChoice']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Random Forest Classifier model
choice_model = RandomForestClassifier(n_estimators=100, random_state=42)
choice_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_choice = choice_model.predict(X_test)
print(classification_report(y_test, y_pred_choice))


### Model Building for Predicting Employment and Financial Outcomes

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Add the predicted education choice to features
X_train['PredictedChoice'] = choice_model.predict(X_train)
X_test['PredictedChoice'] = choice_model.predict(X_test)

# Encoding PredictedChoice
X_train = pd.get_dummies(X_train, columns=['PredictedChoice'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['PredictedChoice'], drop_first=True)

# Predicting Employment Rate
y_train_employment = data.loc[X_train.index, 'EmploymentRate']
y_test_employment = data.loc[X_test.index, 'EmploymentRate']

employment_model = RandomForestRegressor(n_estimators=100, random_state=42)
employment_model.fit(X_train, y_train_employment)

employment_predictions = employment_model.predict(X_test)
print(f'Employment Rate - RMSE: {np.sqrt(mean_squared_error(y_test_employment, employment_predictions))}')
print(f'Employment Rate - R2 Score: {r2_score(y_test_employment, employment_predictions)}')

# Predicting Expected Salary
y_train_salary = data.loc[X_train.index, 'ExpectedSalary']
y_test_salary = data.loc[X_test.index, 'ExpectedSalary']

salary_model = RandomForestRegressor(n_estimators=100, random_state=42)
salary_model.fit(X_train, y_train_salary)

salary_predictions = salary_model.predict(X_test)
print(f'Salary - RMSE: {np.sqrt(mean_squared_error(y_test_salary, salary_predictions))}')
print(f'Salary - R2 Score: {r2_score(y_test_salary, salary_predictions)}')


### Insights Generation and Visualization

# Visualize distribution of forecasted education choices
sns.countplot(y_pred_choice)
plt.title('Distribution of forecasted Education Choices')
plt.show()

# Visualize forecasted employment rate vs actual
plt.scatter(y_test_employment, employment_predictions)
plt.plot([50, 100], [50, 100], '--', color='red')
plt.xlabel('Current Employment Rate')
plt.ylabel('Forecasted Employment Rate')
plt.title('Current vs Forecasted Employment Rate')
plt.show()

# Visualize forecasted salary vs actual
plt.scatter(y_test_salary, salary_predictions)
plt.plot([30000, 80000], [30000, 80000], '--', color='red')
plt.xlabel('Current Salary')
plt.ylabel('Forecasted Salary')
plt.title('Current vs Forecasted Salary')
plt.show()

