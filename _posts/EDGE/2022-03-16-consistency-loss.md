---
title: "[Survey] Consistency Loss"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Edge Learning
tags:
  - []
---

COnsistency loss 전반에 대한 정리글이다.


# Adaptive Network Alignment with Unsupervised and Multi-order Convolutional Networks

[링크](https://ieeexplore.ieee.org/abstract/document/9101653)에서 확인할 수 있다.

해당 논문에서는 GCN 모델 구조를 사용하고 있고, graph domain이라는 점에 유의하여 보자. 

우선 graph데이터를 다루기 때문에 몇가지 가정이 있는데 그 중 homophily가 이해하는데에 핵심이 된다. 결국 homophily라른 것이 network의 구조(structure)가 일관성이 있어야 한다는 아이디어라 생각한다면,<br>
GCN에서 각 layer의 output이 가지고 있는 structure도 일관성있어야 한다는 얘기가 된다. 

GCN에서는 layer가 늘어날수록, 결국 adjacency matrix를 더 곱하는 형태가 되니 낮은 레이어에선 비교적 local한 정보를, 깊은 layer에서는 보다 global한 정보를 가지고 있는 형태의 embedding을 반환하게 된다.<br>
그렇기 때문에 낮은 레이어에서는 서로 다른 node의 embedding이더라도 local구조가 비슷하다면 서로 다른 node도 비슷한 embedding을 가질 수 있게 되고<br>
layer가 깊은 경우 너무 광범위한 정보를 가지고 있는 형태가 되어서 또 비슷비슷하게 생길 수 있게 된다. 

그래서 이렇게 얻은 embedding의 구조가 Laplacian matrix 형태와 유사하도록 만드는 것을 Consistency loss로 사용한다.