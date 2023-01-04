---
title: "[Paper] Spectral Collaborative Filtering"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Recommendation-System
tags:
  - []
---

"Spectral Collaborative Filtering"이란 논문에 대한 리뷰입니다.

원문은 [링크](https://dl.acm.org/doi/abs/10.1145/3240323.3240343?casa_token=526pHPlSJwwAAAAA:DxlSgXqpo9DLNj4HfomsF2BZkWM3ZammzkUmJbCX1_wxq-9h_6k6MwQGKeDJ5m42BHe5S4thO-I)에서 확인할 수 있습니다.

graph domain에서 생각해보면 recommentdation은 bipartite한 경우를 생각해볼 수 있다. 

cold start problem을 지적하면서 graph domain에선 connectivity information을 model에 반영하는 것이 중요하다. <br>
그래서 spectral graph theory를 이용하는데 왜냐하면 eigenvalue(spectrum)이 graph의 proximity보다 connectivity를 반영하기 때문이다. 
- 즉 vertices를 frequency domain에 올리고 이 때 사용되는 basis가 laplacian matrix의 eigenvector다. 
- 즉 connectivity를 반영하기 위함이다. 

graph의 signal을 각 node별 scalar 값으로 생각하면 하나의 vector가 되고 fourier transform matrix를 U matrix라고 하자. <br> 
U는 orthogonal하니 U^T를 사용하면 inverse transform이 되어서 다시 되돌릴 수 있는데 이렇게 다시 되돌리는 것을 일종의 convolution  filter가 된다.<br> 
원래 diag(e.value)를 사용하는데 여기에 weight를 주어서 parameterize를 시켜 learnable parameter로 만든다. 

이렇게 하면 2가지 문제로 한 노드의 feature를 scalar로 제한한다는 단점이 있고 time complexity 문제가 있다.
- time complexity의 경우 polynomial approximation을 통해 해소할 수 있고, polynomial order가 hyperparameter 된다. 
- scalar 문제는 parallel하게 진행하여 input channel을 늘린다면 해결이 가능하고 그래서 "CN"이 된다.
