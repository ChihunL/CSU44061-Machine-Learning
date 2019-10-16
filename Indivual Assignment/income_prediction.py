import csv
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import category_encoders as ce
from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    #Note: This program is non-deterministic as it uses RandomForestRegressor
    #If you wish to make the program deterministic set random_state parameter in
    #RandomForestRegressor to a specific integer

    with open('tcd ml 2019-20 income prediction training (with labels).csv', 'r') as f:
        train_y = list(csv.reader(f, delimiter=','))
    train_y.pop(0)
    train_y = np.array(train_y)
    y = train_y[:, np.newaxis, -1].astype(np.float)
    train_data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
    X = train_data.drop('Income in EUR', axis=1)

    X['Gender'].replace('0', '#N/A')
    X['Gender'].replace('unknown', '#N/A')
    X['Hair Color'].replace('0', '#N/A')
    X['Hair Color'].replace('Unknown', '#N/A')

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        # ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        ('target', TargetEncoder())])
    # ('weightoe', ce.one_hot.OneHotEncoder())])

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    """
    @@@Code used to find best_params

    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.125)
    param_grid = {
        'regressor__n_estimators': [100,200,500],
        'regressor__max_features': ['auto', 'sqrt', 'log2'],
        'regressor__max_depth': [1, 2, 4, 6, 8],
        'regressor__random_state': [randSeed]
    }
    CV = GridSearchCV(reg, param_grid, n_jobs=5)
    CV.fit(train_x, train_y.flatten())
    print(CV.best_params_)
    """

    reg = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=500))])

    train_x = X
    train_y = y
    reg.fit(train_x, train_y.flatten())
    test_x = pd.read_csv(
        'tcd ml 2019-20 income prediction test (without labels).csv')
    test_x['Gender'].replace('0', '#N/A')
    test_x['Gender'].replace('unknown', '#N/A')

    test_x['Hair Color'].replace('0', '#N/A')
    test_x['Hair Color'].replace('Unknown', '#N/A')
    y_pred = reg.predict(test_x)
    print(y_pred)
    writeAnswer(y_pred)

#Prints and returns the mean_squared_error
def getScore(regr, test_y, y_pred):
    # Mean squared error
    mse =  np.sqrt(mean_squared_error(test_y, y_pred))
    print("Root Mean squared error: %.2f"
          % mse)
    return mse

#Function that writes predictions to file
def writeAnswer(y_pred):
    i = 111994
    res = 'Instance,Income\n'
    for x in y_pred:
        res = res + str(i) + ',' + str(x) + '\n'
        i = i + 1
    with open('tcd ml 2019-20 income prediction submission file.csv', 'w') as f:
        f.writelines(res)

if __name__ == "__main__":
    main()
