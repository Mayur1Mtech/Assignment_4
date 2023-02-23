import inline
import matplotlib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.lr_model = LinearRegression()
        self.dt_model = DecisionTreeRegressor(random_state=42)
        self.rf_model = RandomForestRegressor(random_state=42)
        self.lr_param_grid = {'fit_intercept': [True, False]}
        self.dt_param_grid = {'max_depth': np.arange(1,11)}
        self.rf_param_grid = {'n_estimators': [10, 50, 100, 200], 'max_depth': np.arange(1,11)}


    def load_data(self):
        self.data = pd.read_csv(self.data_file_path)

    def perform_eda(self):
        print("Data Info:\n", self.data.info())
        print("Data Shape:\n", self.data.shape)
        print("Data Description:\n", self.data.describe())
        print("Missing Value: \n", self.data.isna().sum())
        print("Outliers: \n", self.data[["Hours"]].boxplot())
        print("Correlation matrix: \n", self.data.corr())
        corr_coef = np.corrcoef(self.data['Hours'], self.data['Scores'])[0, 1]
        print("Correlation coefficient:", corr_coef)
        print()

    def split_data(self, test_size=0.2, random_state=42):
        x = self.data[['Hours']]
        y = self.data['Scores']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size,
                                                                                random_state=random_state)

    def train_models(self):
        # perform grid search to find the best hyperparameters for each model
        lr_grid_search = GridSearchCV(self.lr_model, self.lr_param_grid, cv=5, scoring='neg_mean_squared_error')
        dt_grid_search = GridSearchCV(self.dt_model, self.dt_param_grid, cv=5, scoring='neg_mean_squared_error')
        rf_grid_search = GridSearchCV(self.rf_model, self.rf_param_grid, cv=5, scoring='neg_mean_squared_error')

        # fit the models with the best hyperparameters and make predictions on the testing data
        lr_grid_search.fit(self.X_train, self.y_train)
        lr_pred = lr_grid_search.predict(self.X_test)

        dt_grid_search.fit(self.X_train, self.y_train)
        dt_pred = dt_grid_search.predict(self.X_test)

        rf_grid_search.fit(self.X_train, self.y_train)
        rf_pred = rf_grid_search.predict(self.X_test)

        # evaluate the models
        lr_rmse = np.sqrt(mean_squared_error(self.y_test, lr_pred))
        lr_r2 = r2_score(self.y_test, lr_pred)
        print('Linear Regression - RMSE:', lr_rmse)
        print('Linear Regression - R-squared:', lr_r2)
        print()

        dt_rmse = np.sqrt(mean_squared_error(self.y_test, dt_pred))
        dt_r2 = r2_score(self.y_test, dt_pred)
        print('Decision Tree - RMSE:', dt_rmse)
        print('Decision Tree - R-squared:', dt_r2)
        print()

        rf_rmse = np.sqrt(mean_squared_error(self.y_test, rf_pred))
        rf_r2 = r2_score(self.y_test, rf_pred)
        print('Random Forest - RMSE:', rf_rmse)
        print('Random Forest - R-squared:', rf_r2)
        print()


if __name__ == '__main__':
    trainer = ModelTrainer("/home/mayur/Desktop/Assignment_4/DATA.csv")
    trainer.load_data()
    trainer.perform_eda()
    trainer.split_data()
    trainer.train_models()
