---
title: "[Paper] Optimization as a Model for Few-Shot Learning "
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Self Supervised
tags:
  - [survey]
---

"Optimization as a Model for Few-Shot Learning "이란 논문에 대한 리뷰입니다.

원문은 [링크](https://openreview.net/forum?id=rJY0-Kcll)에서 확인할 수 있습니다.

# Idea
기존의 gradient update eq가 LSTM의 cell state update eq과 동일하다. 
 - model은 model 대로 돌리고, update를 lstm처럼 진행한다. 

- 목적은 비슷한 데이터 셋들을 골고루 학습하여 generalizability를 향상시키는 것이다. 즉 shot은 class별 label 개수가 해당하는 것이라 few shot learning이 된다. 

# LSTM과 비교
Cell state는 parameter를 <br>
Input gate는 learning rate를<br>
Forget gate는 1을,<br>
Input candidate(gt)는 gradient로 생각할 수 있다는 것. <br>

# 적용
- input gate와 forget gate를 각 이전 step의 gate 값과 paremeter값, Loss, gradient를 이용하여 학습시킨다. 
 - Forget gate를 1로 하는게 이상한거

- 중요한 점은 cell state(parameter)를 얻고자 하는 것이지 hidden state(ht)를 얻는 게 아니니 ht와 관련된 건 계산하지 않는다. 이 부분과 관련해선 LSTM의 original paper에 식을 따르는 개념이다. 
 - 여기엔 hidden state 개념이 없으니

- 단순히 optimizer를 meta learner라고, model을 learner라고 지칭하며 learner는 loss와 gradient를, meta learner는 parameter update를 제공하는 것. 

- 특징
 - 개별 데이터 셋을 episode라고 호칭
 - Test set에 관해서도 meta learner는 학습하지만 learner는 학습하지 않는다.
