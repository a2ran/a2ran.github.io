---
title: "서울시 자전거 대여 데이터셋 회귀와 분류"
toc: true
use_math: true
categories:
  - app
tags:
  - [Machine Learning, Regression, classification, projects]
date: 2023-02-05
last_modified_at: 2023-02-05
sitemap:
  changefreq: daily
  priority: 1.0
---

서울시 따릉이 공공데이터를 받아와 자전거 대여 회귀분석과 비오는날 분류를 진행합니다.

## 칼럼설명

* id : 마포구에 있는 따릉이 보관소의 고유 id
* hour : 따릉이 보관소에서 기상상황을 측정한 시간
* temperature : 기온
* precipitation : 비가 오지 않았으면 0, 비가 오면 1, null은 비가 온 것도 아니고 안 온 것도 아니라서 센서가 확실히 측정불가한 상태
* windspeed : 풍속(평균)
* humidity : 습도
* visibility : 시정(視程), 시계(視界)(특정 기상 상태에 따른 가시성을 의미)
* ozone : 오존
* pm10 : 미세먼지(머리카락 굵기의 1/5에서 1/7 크기의 미세먼지)
* pm2.5 : 미세먼지(머리카락 굵기의 1/20에서 1/30 크기의 미세먼지)
* count : 측정한 날짜의 따릉이 대여 수

## 사용 모델 정리

* DecisionTree(의사결정나무)
    * DecisionTree : DecisionTreeClassifier
    * RegressionTree(회귀나무) : DecisionTreeRegressor
* Ensemble(앙상블)
    * Voting : VotingClassifier & VotingRegressor
        * VotingClassifier 참고자료
          https://yganalyst.github.io/ml/ML_chap6-1/

        * VotingRegressor 참고자료
          1. https://runebook.dev/ko/docs/scikit_learn/modules/generated/sklearn.ensemble.votingregressor
          2. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html#sklearn.ensemble.VotingRegressor
    * Bagging : RandomForestClassifier & RandomForestRegressor
    * Boosting
        * GBM : GradientBoostingClassifier & GradientBoostingRegressor
        * XGBoost : XGBClassifier & XGBRegressor
        * LightGBM : LGBClassifier & LGBRegressor
        * CatBoost : CatBoostClassifier & CatBoostRegressor
            * CatBoost 참고자료 : https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier
    * Stacking : https://lsjsj92.tistory.com/558

## 코드

해당 링크에 들어가 계산과정을 확인 가능합니다.

[Github Page Link](https://github.com/a2ran/ml-rl-applications/blob/main/%5B0202%5D_DecisionTree_and_Ensemble_UiHyunCho.ipynb)

## 결과

### Regressor

> R square score for RegressionTree model : 0.654 <br>
> R square score for RandomForest model : 0.769 <br>
> R square score for XGBoost model : 0.782 <br>
> R square score for LightGBM model : 0.768

## Classifier

(Accuracy Score)

> DecisionTree model : 0.966
> RandomForest model : 0.981
> XGBoost model : 0.981
> LightGBM model : 0.981
