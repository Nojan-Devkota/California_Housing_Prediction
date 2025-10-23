import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. load dataset
housing = pd.read_csv("Cali_housing_predict/housing.csv")

# 2. creating stratified data set based on income category
# (to ensure every category is included in the splitted dataset)

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

split = StratifiedShuffleSplit(n_splits=1, test_size= .2, random_state= 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis = 1)
    strat_test_set = housing.loc[test_index].drop('income_cat', axis = 1)

#3. Working on the copy of the training data
housing = strat_train_set.copy()

#Separate features and labels
housing_labels = housing['median_house_value'].copy()
housing_features = housing.drop('median_house_value', axis = 1).copy()

#4. Separate numerical and categorical columns
num_attributes = housing_features.drop('ocean_proximity', axis = 1).columns.tolist()
cat_attributes = ['ocean_proximity']

#5. Making Pipelines
#Numerical Pipeline

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

#Categorical Pipeline
cat_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

#Full pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes)
])

#6. Transform the data
housing_features_transformed = full_pipeline.fit_transform(housing_features)
housing_features_transformed = pd.DataFrame(
    housing_features_transformed,
    columns= full_pipeline.get_feature_names_out(),
    index= housing_features.index
)

#7. Train the model
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_features_transformed, housing_labels)
lin_pred = lin_reg.predict(housing_features_transformed)
lin_rsme = root_mean_squared_error(housing_labels, lin_pred)
print(f'RSME of Linear Regression: {lin_rsme}')

lin_cross_val = -cross_val_score(lin_reg, housing_features_transformed, housing_labels, cv=10, scoring='neg_root_mean_squared_error')
print(f"Cross Val Error of Linear Regression: {lin_cross_val.mean()}\n")

#Decision Tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_features_transformed, housing_labels)
tree_pred = tree_reg.predict(housing_features_transformed)
tree_rsme = root_mean_squared_error(housing_labels, tree_pred)
print(f"RSME of Decision Tree: {tree_rsme}")

tree_cross_val = -cross_val_score(tree_reg, housing_features_transformed, housing_labels, cv=10, scoring='neg_root_mean_squared_error')
print(f"Cross Val Error of Decision Tree: {tree_cross_val.mean()}\n")

#Random Forest 
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_features_transformed, housing_labels)
random_forest_pred = random_forest_reg.predict(housing_features_transformed)
random_forest_rsme = root_mean_squared_error(housing_labels, random_forest_pred)
print(f"RSME of Random Forest: {random_forest_rsme}")

random_forest_cross_val = -cross_val_score(random_forest_reg, housing_features_transformed, housing_labels, cv=10, scoring='neg_root_mean_squared_error')
print(f"Cross Val Error of Random Forest: {random_forest_cross_val.mean()}\n")