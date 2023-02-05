---
title: "서울시 부동산 데이터 보증금 회귀분석"
toc: true
use_math: true
categories:
  - app
tags:
  - [Machine Learning, Regression, projects]
date: 2023-02-05
last_modified_at: 2023-02-05
sitemap:
  changefreq: daily
  priority: 1.0
---

네이버 부동산에서 크롤링해 가져온 서울시 부동산 데이터를 전처리 후 회귀분석을 통해 예측합니다.

## 칼럼설명

* Unnamed: 0 : 의미없는 칼럼
* id : 매물id
> 한 매물당 하나의 id가 할당된다.
* lat : 위도
* lng : 경도
* gu : 구(행정지역정보)
* goodsType : 매물의 거래형태(아파트,빌라 등)
* payType : 판매의 거래형태(월세,매매 등)
* floor : 매물방의 층수
* floor_total : 매물이 있는 전체 건물의 층수
* contractArea : 계약면적
* realArea : 실제면적
* direction : 창문방향(남향,북향)
* tag : 매물마다 관련된 설명정보
> ex. 일조량 및 관리 잘 된 복층
* tagList : 매물마다 관련된 설명에서의 태그
> ex. '10년이내, 25년이내 건축, 복층'과 같이 인스타 해시테그 느낌
* deposit : 보증금
* monthlyPay : 월세


> 데이터 출처 : 네이버 부동산 크롤링 데이터

## 코드

해당 링크에 들어가 계산과정을 확인 가능합니다.

###[Github Page Link](https://github.com/a2ran/ml-rl-applications/blob/main/%5B0131%5D_LR_UiHyunCho.ipynb)

## 결과

> R^2 Ridge score : 0.5299 <br>
> R^2 Lasso score : 0.5299 <br>
> MSE Ridge score : 0.4957 <br>
> MSE Lasso score : 0.4956

<img src = '/assets/images/app/1.png'>

## 결론

> Ridge와 Lasso 규제는 거의 동일한 효과를 나타냈다 <br>
> $R^2$ 값은 0.53이 나왔지만 MSE, MAE, RMSE가 아주 적은 값이 나왔다..!!<br>
> one-hot encoding쪽에서 다중공산성 분석으로 조금 다듬으면 더 높은 R2score값을 기대 가능합니다.
