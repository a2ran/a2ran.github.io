---
title: "[Paper] Generative Adversarial Nets"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Generative
tags:
  - []
---

Generative model 중 흔히 언급되는 모델들을 정리한 글입니다.

# Energy based Model
Density function을 추정하기 위해 energy function의 negative exponential 값을 사용하고 normalize를 취한 것을 사용한다. 그렇기에 좋은 점이라고 하면 prior assumption이 없다.

이 때 energy function이란, realistic point은 적은 값을, unrealistic point는 큰 값을 가지게 하는 함수다. 
 - Likelihood에 비례하는 값이라서 결국 계산하기가 intractable하다. 그래서 MCMC 같은 방법이 제안되며 다른 방식으로 확장된다. 

이런 문제를 해결하기 위해 proxy objective인 contrastive divergence를 사용하게 된다. 이를 통해 data sample에 대한 energy 값은 낮아지고 generated sample은 높아지게 된다. 이건 결국 negative likelihood를 사용하는 것이고, 실제 데이터에서 나온 것은 증가하는 gradient, estimate 분포에서 generated sample은 감소하는 gradient가 된다. 
 - 이게 결국 Generative Model에서의 철학으로 생각한다. 실제 데이터 sample와 estimator의 sample별로 optimization 방향을 달리하여 학습하는 것

결국 MCMC를 한다는 것 자체가 intractable한 특징을 circumvent하는 것 뿐.
 
# Boltzmann Machine
Fully connected undirected network이며 각각의 node(neuron)들은 input에 대해 weight sum하고 sigmoid를 취한 확률 값을 반환하는 machine이다. 

이 또한 Energy function으로 정의할 수 있고, 이것도 Energy based Model처럼 loss를 정의할 수 있다.  하지만 이 방법 또한 energy function의 gradient를 계산하기에 연산량이 많아 Gibbs sampling이 사용되기도 하는데 결국 low dimensional에만 사용이 가능하다. 

Cf) Restricted Boltzmann Machine
Fully connectedness를 포기한 BM이다. 
 
# Variational AE
Latent variable, Z를 이용하여 P(X|Z)와 P(Z|X)를 유사하도록 학습한다. 즉 KL Divergence를 정의할 수 있게 되고, 식변환을 통해서 P(X)의 lower bound를 계산할 수 있게 되며 이것을 Evidence Lower Bound(ELBO)라고 한다. 

단 blurry하고 unrealistic하다는 단점이 있다. 
 
