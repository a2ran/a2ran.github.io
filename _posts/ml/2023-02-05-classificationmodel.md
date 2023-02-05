---
title: "GridSearch Classification Models"
toc: true
use_math: true
categories:
  - ml
tags:
  - [Machine Learning, Classification, Math]
date: 2023-02-05
last_modified_at: 2023-02-05
sitemap:
  changefreq: daily
  priority: 1.0
---

Optimize classification problem by gridsearching through several models and hyperparameters.

## Prerequisites

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
```

## 1. LogisticRegression

```python
from sklearn.linear_model import LogisticRegression

# Define machine learning model
lr = LogisticRegression()

# Fit the model
lr.fit(train_X, train_y)

# Define Hyperparameters
parameters = {'C' : [0.01, 0.1, 1, 10],
             'random_state' : [1]}

# Fit the model using optimized parameters
model_lr = GridSearchCV(estimator = lr, param_grid = parameters,
                       scoring = 'f1', cv = 10, refit = True)
model_lr.fit(train_X, train_y)
```

## 2. SupportVectorMachine

```python
from sklearn.linear_model import SVC

svm = SVC()
svm.fit(train_X, train_y)

parameters = {'kernel' : ['rbf', 'linear', 'poly'],
             'C' : [0.5, 1.5, 10],
             'random_state' : [1]}

model_svm = GridSearchCV(estimator = model_svm, param_grid = parameters,
                        scoring = 'recall', cv = 10, refit = True)
model_svm.fit(train_X, train_y)
```

## 3. DecisionTreeClassifier

```python
from sklearn.tree import DecisionTreeClassifier

clf_tree = DecisionTreeClassifier()

clf_tree.fit(train_X, train_y)

parameters={'criterion':['gini', 'entropy'],
            'max_depth':[5,10,15,20,None]}

model_dt=GridSearchCV(estimator=clf_tree, param_grid=parameters,
                     scoring='f1', cv=10, refit=True)

model_dt.fit(train_X,train_y)
```

## 4. RandomForestClassifier

```python
randomforest=RandomForestClassifier()

randomforest.fit(train_X, train_y)

parameters = {'max_depth' : [3, 5, 10],
             'n_estimators' : [100, 200, 300],
             'random_state' : 1}

model_rfc=GridSearchCV(estimator = randomforest, param_grid = parameters,
                      scoring = 'f1', cv = 10, refit = True)

model_rfc.fit(train_X,train_y)
```

## 5. XGBClassifier

```python
from xgboost import XGBClassifier

xgboost=XGBClassifier()

xgboost.fit(train_X, train_y)

parameters = {'max_depth' : [3, 5, 10],
             'n_estimators' : [100, 200, 300],
             'learning_rate' : [0.001, 0.01, 0.1, 1],
             'gamma' : [0.5, 1, 2],
             'random_state' : [1]}

model_xgb = GridSearchCV(estimator = xgboost, param_grid = parameters,
                        scoring = 'f1', cv = 10, refit = True)

model_xgb.fit(train_X, train_y)
```

## 6. LGBMClassifier

```python
from lightgbm import LGBMClassifier

# 객체 선언
lightgbm=LGBMClassifier()

lightgbm.fit(train_X, train_y)

parameters = {'max_depth' : [3, 5, 10],
             'n_estimators' : [100, 200, 300],
             'learning_rate' : [0.001, 0.01, 0.1, 1],
             'random_state' : [1]}

model_lgb = GridSearchCV(estimator = lightgbm, param_grid = parameters,
                        scoring = 'f1', cv = 10, refit = True)

model_lgb.fit(train_X, train_y)
```

## 7. Model Evaluation

### f1-score

```python
print("DecisionTree model : {:.3f}".format(f1_score(train_y, model_dt.predict(train_X))))
print("RandomForest model : {:.3f}".format(f1_score(train_y, model_rfc.predict(train_X))))
print("XGBoost model : {:.3f}".format(f1_score(train_y, model_xgb.predict(train_X))))
print("LightGBM model : {:.3f}".format(f1_score(train_y, model_lgb.predict(train_X))))
```

### accuracy score

```python
print("DecisionTree model : {:.3f}".format(model_dt.best_score_))
print("RandomForest model : {:.3f}".format(model_rfc.best_score_))
print("XGBoost model : {:.3f}".format(model_xgb.best_score_))
print("LightGBM model : {:.3f}".format(model_lgb.best_score_))
```

### precision score

```python
print("DecisionTree model : {:.3f}".format(precision_score(test_y, model_dt.predict(test_X))))
print("RandomForest model : {:.3f}".format(precision_score(test_y, model_rfc.predict(test_X))))
print("XGBoost model : {:.3f}".format(precision_score(test_y, model_xgb.predict(test_X))))
print("LightGBM model : {:.3f}".format(precision_score(test_y, model_lgb.predict(test_X))))
```
