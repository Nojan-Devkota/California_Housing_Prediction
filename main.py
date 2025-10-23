import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import root_mean_squared_error
# from sklearn.model_selection import cross_val_score

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attributes, cat_attributes):
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
    
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # 1. load dataset
    housing = pd.read_csv("housing.csv")

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
        housing.loc[test_index].drop(['income_cat', 'median_house_value'], axis = 1).to_csv('input.csv')
        housing.loc[test_index].drop(['income_cat'], axis = 1).to_csv('input_copy.csv')

    #3. Working on the copy of the training data
    housing = strat_train_set.copy()

    #Separate features and labels
    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis = 1).copy()

    #4. Separate numerical and categorical columns
    num_attributes = housing_features.drop('ocean_proximity', axis = 1).columns.tolist()
    cat_attributes = ['ocean_proximity']

    pipeline = build_pipeline(num_attributes, cat_attributes)
    housing_features_transformed = pipeline.fit_transform(housing_features)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_features_transformed, housing_labels)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is trained. Congrats!!")
        
else:
    #Lets do inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    input_data = pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value_predictions'] = predictions
    
    input_data.to_csv('output.csv', index= False)
    print("Inference is complete, results saved to output.csv")