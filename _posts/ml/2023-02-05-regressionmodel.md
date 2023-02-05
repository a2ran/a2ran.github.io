---
title: "GridSearch Regression Models"
toc: true
use_math: true
categories:
  - ml
tags:
  - [Machine Learning, Regression, Math]
date: 2023-02-05
last_modified_at: 2023-02-05
sitemap:
  changefreq: daily
  priority: 1.0
---

Optimize regression problem by gridsearching through several models and hyperparameters.

## Prequisites

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
```

## 1. DecisionTreeRegressor

```python
from sklearn.tree import DecisionTreeRegressor

reg_tree = DecisionTreeRegressor() # Regression Model

reg_tree.fit(train_X, train_y)

# HyperParameters
parameters={'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_depth':[5,10,15,20,None]}

# Tune Hyperparameters
model_rt=GridSearchCV(estimator=reg_tree, param_grid=parameters,
                     scoring='r2', cv=10, refit=True)

# Train model using optimized parameters
model_rt.fit(train_X,train_y)
```

## 2. RandomForestRegressor

```python
from sklearn.ensemble import RandomForestRegressor

randomforest=RandomForestRegressor()

randomforest.fit(train_X, train_y)

parameters={'max_depth':[3,5,10],
            'n_estimators':[100,200,300],
            'random_state':[1]}

model_rfr=GridSearchCV(estimator=randomforest, param_grid=parameters,
                 scoring='r2', cv=10, refit=True)

model_rfr.fit(train_X,train_y)
```

## 3. XGBRegressor

```python
from xgboost import XGBRegressor

xgboost=XGBRegressor(objective ='reg:squarederror')

xgboost.fit(train_X, train_y)

parameters={'max_depth':[3,5,10],
            'n_estimators':[100,200,300],
            'learning_rate':[1e-3,0.01,0.1,1],
            'gamma':[0.5,1,2],
            'random_state':[1]}

model_xgb=GridSearchCV(estimator=xgboost, param_grid=parameters,
                     scoring='r2', cv=10, refit=True)

model_xgb.fit(train_X,train_y)
```

## 4. LGBMRegressor

```python
from lightgbm import LGBMRegressor

lightgbm=LGBMRegressor()

lightgbm.fit(train_X, train_y)

parameters={'max_depth':[3,5,10],
            'n_estimators':[100,200,300],
            'learning_rate':[1e-3,0.01,0.1,1],
            'random_state':[1]}

model_lgbm=GridSearchCV(estimator=lightgbm, param_grid=parameters,
                     scoring='r2', cv=10, refit=True)

model_lgbm.fit(train_X,train_y)
```

## 5. Model Evaluation

### Mean Squared Error

```python
print("RegressionTree model : {:.3f}".format(mean_squared_error(train_y, model_rt.predict(train_X))))
print("RandomForest model : {:.3f}".format(mean_squared_error(train_y, model_rfr.predict(train_X))))
print("XGBoost model : {:.3f}".format(mean_squared_error(train_y, model_xgb.predict(train_X))))
print("LightGBM model : {:.3f}".format(mean_squared_error(train_y, model_lgbm.predict(train_X))))
```

### R^2 Score

```python
print("R square score for RegressionTree model : {:.3f}".format(model_rt.best_score_))
print("R square score for RandomForest model : {:.3f}".format(model_rfr.best_score_))
print("R square score for XGBoost model : {:.3f}".format(model_xgb.best_score_))
print("R square score for LightGBM model : {:.3f}".format(model_lgbm.best_score_))
```
