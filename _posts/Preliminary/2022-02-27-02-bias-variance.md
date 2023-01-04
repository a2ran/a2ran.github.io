---
title: "[Note] Bias & Variance"
toc: true
use_math: true
categories:
  - Preliminary
---

Bias와 Variance간 trade off 혹은 그 decomposition에 대한 정리글입니다.

# Assumption

We hope to minimize training error so that the "generalization error" getting lower, too.<br>
That's why we need to know where the gernalization data come from

$D = \lbrace (x_{1}, y_{1}), ... \rbrace , y \in R$ from iid $P(X,Y)$

# Definition

Expected lable $\bar{y}$ : 데이터 x가 주어졌을 때 y값($y\mid x$)에 대한 기대값으로 y에 대한 integral<br>
$h_{D}$ : 데이터 D를 입력 후 알고리즘 A를 거쳐 나온 결과<br>
Expected test error given $h_{D}$ : 실제 결과값 y와 $h_{D}$의 에러에 대한 기대값으로 x와 y에 대한 double integral<br>
Expected classifier $\bar{h}$ : $h_{D}$의 기댓값으로 $D$에 대한 integral<br>
Expected error of A : 실제 결과값 y와 $h_{D}$의 에러에 대한 기댓값으로 test error와 동일하게 생각할 수 있지만 **x와 y와 D**에 대한 integral<br>

위의 정의에서 중요한 것은 우리가 실제 테스트를 해보고 나오는 에러와 모델의 에러를 구분할 줄 알아야 한다는 것입니다.

위의 정의에선 X,Y 그리고 $D$를 구분하여 다루고 있으며 test error의 경우에는 $D$에 대한 integral이 없기 때문에 $D$에 따라서 test error가 바뀔 수 있다는 점이 되고,<br>
그렇기 때문에 어떤 모델의 에러와 차이가 생기게 되어서 결국 **Generalization**에는 결국 $D$에 의한 영향도 고려하게 됩니다.

# Decomposition

기존의 Bias-Variance trade-off에서 하던 대로 $h_{D}$와 y의 Squared Error를 X,Y 그리고 $D$에 대해서 기댓값을 계산해보도록 합시다.

$\bar{h}$를 추가하여 이차식을 전개해볼 경우 
1. $h_{D}$와 $\bar{h}$ 간 error 제곱
2. $\bar{h}$와 y 간 error 제곱
 - 해당 값들은 $D$에 대해선 상수값들이기 때문에 다시 이것을 $\bar{y}$를 이용해서 decomposition을 진행하게 되고 cross term은 0이 됩니다. 그 결과 아래의 항들이 남게 됩니다.
  - $\bar{h}$와 $\bar{y}$ 간 error 제곱
  - $\bar{y}$와 y간 error 제곱
3. 1번과 2번의 cross term
 - cross term은 0이 되는데 그런 이유가 $\bar{h}$는 D에 대해서 이미 integral이 되었기 때문에 상수로 계산이 되어서 $D$에 대해서 $h_{D}$와 $\bar{h}$의 차이의 기댓값이 0이 된 것입니다. 

# Result 

위의 3가지 항을 정리하였을 때 남는 것은 다음과 같고 각각의 의미는 다음과 같습니다.

- $h_{D}$와 $\bar{h}$ 간 error 제곱
 - *Variance* of **Algorithm**, Not about getting right or wrong
- $\bar{h}$와 $\bar{y}$ 간 error 제곱
 - squared *Bias* of **Algorithm**
- $\bar{y}$와 y간 error 제곱
 - *Noise* of **data**, a value that cannot be made better than current Algorithm because $\bar{y}$ is expected one, which is a a kind of label that algorithm think it should have
 - This is natural aspect as a data or observation. No matter how much data have, always existing error.

즉 여기서 중요한 점은 결국 에러라고 하는 것이 크게는 Data와 Alogorithm에서 기인하는 것이고, Alogirhtm의 에러의 경우에는 기존의 Bais-Variance trade off로 생각할 수 있으며 Noise는 더 이상 낮출 수 없는 에러임을 알아야 합니다.<br>
더 나아가, 현재 어떤 에러가 존재한다면 앞서 말한 3가지 에러 중 어디에 속하는지 구분할 줄 알아야 하며 noise는 모델 혹은 알고리즘을 개선한다고 해서 개선할 수 없는 자연적인 것임을 알아야 합니다.
