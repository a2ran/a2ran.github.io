---
title: "[Paper] Communication-Efficient Device Scheduling for Federated Learning Using Stochastic Optimization"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Edge Learning
tags:
  - []
---

"Communication-Efficient Device Scheduling for Federated Learning Using Stochastic Optimization"이란 논문에 대한 리뷰입니다.

원문은 [링크](https://arxiv.org/abs/2201.07912)에서 확인할 수 있습니다.

# Abstract
- Convergence analysis
- (In common) Stochastic optimization for a new client selection
- (In distinct) Minimization of average communcation time 

# Introduction
- (In common) consumption of more network resources than necessary
- (In common) consideration abouth the effect of arbitrary device selection on "FL" convergence
 - arbitrary device selection이라고 하는 것이 우리 환경에서의 Random selection으로 생각할 수 있다면 참고해도 될 페이퍼라고 생각
- (In distinct) FL 환경에서 다뤄서 analysis에서 round를 고려하기도 한다. 
- (In disticnt) bottleneck 문제를 communication time을 최소화해서 해결하고자 한다. 

# Problem Formulation
- (In common) client selection과정에서 뽑힐 확률을 조정한다.
 - Likelihood가 큰 것을 고르도록 한다는 점에서 임의로 뽑는 것과 차이가 생기고 이런 아이디어는 본 페이퍼와 DI와 동일하지만, DI에서는 argmax를 고르기에 젤 좋은 디바이스가 뽑힐 확률이 1이고 나머지는 0이지만 본 페이퍼에서는 채널 환경이 안좋아도 뽑힐 확률이 0이 되지는 않는 것 같다.(그래서 Stochastic optimization이라고 얘기하는 것인가?)
- (In distinct) 단 뽑힐 확률을 조정할 때 channel condition과 같은 system dynamics에 따라서 조정된다.
 - 확률을 줄 때 weight를 DI가 아닌 channel condition으로 준다는 점에서 차이


# Convergence Analysis
- (In distinct) 기존에 FL에서 convergence analysis하는 것과 비슷한 맥락으로 진행되고, 결론은 convergence bound가 뽑힐 확률의 함수가 되어 뽑힐 확률을 어떻게 조정해주는지에 따라 convergence bound가 바뀌는 과정을 얘끼해줌 

# Communication-efficient Scheduling Policy
- (In common) 메트릭을 사용해서 디바이스가 뽑힐 확률을 계산한다.
- (In distinct) 메트릭이 통신에서 사용하는 것(CSI)이며 
- (In distinct) 디바이스 스스로가 그 확률을 계산한다는 점에서 우리는 서버에서 계산하기에 차이점이 존재한다. 

이후의 내용들은 통신 분야에 관한 것으로 생략.
