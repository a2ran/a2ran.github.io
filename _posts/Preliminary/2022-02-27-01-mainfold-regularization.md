---
title: "[Note] Manifold Regularization & Laplace Norm"
toc: true
use_math: true
categories:
  - Preliminary
tags:
  - [Regularization]
---

Maniforld Regularization과 Laplace norm에 대한 정리글입니다.

# Manifold regularization
data가 전체 space를 span하지 않고 subset이기 때문에 regularization이 들어가게 되는 것.
  
Assumption
- function to be learned is smooth

data from different class is distinct

- data는 entire domain에서 generate된 것이 아니라 nonlinear manifold which is subset to domain에서 generated된 것으로 본다.  

그리고 이 manifold의 성질을 정의하는 것이 regularization norm이 되는 것이다. 

이런 걸  Tikhonov regularization이라고 부른다!

결국 학습하는 f는 Reproducing Kernel Hilber space(RKHS)을 mapping하게 되는 것이고, 특징은 f의 norm이 kernel K로 정의된다는 것이다. 이 때 norm_{k}는 RKHS의 특징(complexity)를 보여주는 것이 된다. 

       
# Laplace norm
manifold의 gradient를 이용하여 얼마나 target function이 smooth한지 측정하는 값이 된다.<br>
그리고 smooth하다는 말은 gradient 값이 작은데 작은 Support는 xdml pdf에서 dominant한 영역에 해당한다. <br>
결국 function f의 norm은 확률 값의 변화에 따른 gradient의 크기가 된다. <br>

문제는 어떻게 pdf를 estimate를 할 것인지가 중요하고, graph context에서 고려해보면 거리를 이용하기 때문에 Laplacian matrix를 사용하게 된다. 

distance를 측정하는 W와 한 node에 대해 distance의 합을 나타내는 D를 정의하면, Laplacian matrix = D - W로 정의된다.

이 환경에서 labeled와 unlabeled의 수가 늘면 결국 L은 Laplace-Beltrami operator로 수렴하게 되고, 이는 즉 Manifold의 graident의 divergence값을 말한다.!!! 

즉 데이터가 늘어남에 따라서 Laplacian matrix는 graident의 divergence에 수렴하게 되고 이게 그렇게 된다고...? 여긴 graph theory를 더 공부해보자.!

- Laplacian matrix a.k.a. graph laplacain, kirchooff matrx
- matrix representation of a graph
- D(degree matrix) - A(adjacency matrix)   

이렇게 정의하면 diagonal은 degree 값이, offdiagonal은 adjacent 여부에 따라 -1 or 0이 된다.
