---
title: "[Paper] Progressive Feature Transmission for Split Inference at the Wireless Edge"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Edge Learning
tags:
  - []
---

"Progressive Feature Transmission for Split Inference at the Wireless Edge"이란 논문에 대한 리뷰입니다.

원문은 [링크](https://arxiv.org/abs/2112.07244)에서 확인할 수 있습니다.

# Prior Work
- Edge Learning과 Edge Inference로 크게 나뉜다
- 단, 우리가 하는 것은 Edge Learning에 해당하는 것으로, server와 device가 기존의 모델에서 Split한 것을 각각 담당하는 스키마 

# 방향성
- (차이점) latency requirement을 주요 문제로 삼음
- (공통점) communication bottleneck & communication efficient 방향 

# Key
- (Solution) pruning features according to their heterogeneous importance level
- (Scoring Method) observing the effect of is removal on the inference performance
 - 이렇게 하면 결국 매우 느려질 것으로 생각(Time complexity)
- FTX protocol use multiple shot to achieve specific performance
- (Data Distribution) Given the (total) data distribution
- (Classification model) Linear model
 - 우리는 Logistic을 사용하게 되면서 hyperplane에 sigmoid kernel을 취한 형태로 생각하면 될 것. 학습방식은 동일한 것으로 판단
- (Metric) Inference Uncertainty: entropy 
 - Feature 수가 증가할 수록 감소
 - probability of Posterior distribution getting arise(proper)
 - more peaky(certain) for specific label
- (Metric) Discriminant Gain:(pairwise) symetric KL diverence
 - Arranged result 
 - location이 점점 비슷해져
 - 더 안좋은 feature는 gain이 적어진다. 


Based on "Feature selection", performance gain is available and the number of slots is principle in trade-off relationship. 

c.f. that's why the latency averaged for the number of slots is used
