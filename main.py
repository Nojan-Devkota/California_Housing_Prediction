import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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
