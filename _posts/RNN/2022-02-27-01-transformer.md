---
title: "[Paper] Attention is All you Need"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - RNN
tags:
  - [Recurrent NN]
---

"Attention is All you Need"이란 논문에 대한 리뷰입니다.

원문은 [링크](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)에서 확인할 수 있습니다.

# Background 
-	LSTM, Gated RNN
 - Sequence 길이에 영향을 많이 받는다.

-	Seq2Seq
 - input과 output 전체를 다 반영하려다 보니 distant position에 있는 token간 dependency를 학습하기 어렵다.
 - 또한 단순히 input의 encoder의 output인 context vector만을 가지고 decoder를 돌리니 limited한 상태인 것

# Key
-	Why self-attention
  - 0  Computational complexity per layer
  - Parallelizable computation
  -  Path length between long range dependency

-	Decoder의 en-de attention의 경우 query는 이전 decoder layer의 query를 사용하고 key & value는 encoder의 마지막 layer의 output을 사용
 - 이를 통해 decoding을 할 때마다 input sequence 전체에 대한 정보를 반영하게 된다.

-	Decoder의 self attention의 경우 이전 layer의 현재 position까지에 대한 정보를 반영하는 것
-	Encoder의 self attention의 경우 이전 layer의 모든 position에 대한 정보를 반영하는 것.


# Architecture
-	Encoder
  -  6 Stacks of unit
  -  Residual connection(ResNET처럼)
  -  Multi head attention generates Query, Key Value

-	Decoder
  -  Generates Query with Multi head attention
  -  Again Multi head attention with Key & Value of Encoder and Query of Decoder 

-	Dot product Attention
  -  Sofmax(Q@K)/sqrt(d_k) * V
  -  이 때 dot product을 사용하는 것은 matrix로 다루기에 더 빨라지는 장점이 있따.
  -  D_k로 나누는 것은 dot product는 d_k가 클수록 값이 커져 결국 softmax의 gradient를 작게 만드는 효과를 상쇄하기 위함이다.!!!
  즉 확률 값을 계산할 때 value로 판단하게 되고, 이 때 가중치를 query와 key의 dot product이 되며 이는 query와 key가 유사한 정도를 반영하여 계산한다는 것

query와 key의 embedding dim은 d_k가 되고(dot product을 해야하니) value의 차원의 d_v가 된다. <br>
query는 이전까지의 sequence에 대한 정보를, key는 token별 정보를, value는 token별 중요도(일종의 intensity)를 알려준다.<br>
그 결과 query와 key의 dot product은 이전까지의 sequence과 현재 후보 token의 유사한 정보를 나타내어, token별 중요도에 weight로 계산하여 최종 score가 계산이 되는 것.

-	Multi-Head Attention
  -  동일한 d_model에서 d_k와 d_v로 각각 projection하며 이를 h번 반복하는 것.
  -  즉 h번을 parallel하게 진행하고, concat하여 합친 뒤 value를 계산하게 된다. 
  -  h번을 parallel하게 진행하기 때문에 서로 다른 representation subspace에서 정보를 모으게 된다.
  -  개인적으로 일종의 ensemble과 같은 역할을 하는 것 같다. 

# 요약

N개의 sequence의 element에 대해 각각 d_model의 embedding dim을 가진다. 
-	Q,K,V : N*d_model

이를 각각 Query, Key, Value로 바꿔주는 matrix
-	Wq: d_model*d_k, 
-	Wk: d_model*d_k
-	Wv : d_model*d_v

d_v의 attention의 output이 h개가 있고, 이를 다시 model의 embedding으로 바꿔주는 W_O
-	Wo: hd_v*d_model

최종 output이 d_model이 되어야 stack할 수 있다. 

-	Positional Encoding

Relative or absolute position 정보를 반영하게 되며 동일하게 d_model의 차원을 가지고 summation으로 반영시킨다.
 - 각 차원에 대해 sinusoid한 관계로 정의된다.

그러니까 d_model의 vector를 만들 때 해당 token이 sequence 상에서의 위치에 대한 정보에다가 dim 값들이 홀수는 cos 관계를, 짝수는 sin 관계로 변환시킨 결과를 사용한다. 

# Insight
-	embedding dim이 성능에 끼치는 영향
 - 중간에 dim으로 나눠주는 이유가 된다!
