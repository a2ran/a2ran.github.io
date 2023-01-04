---
title: "[Paper] Local2Global Review"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Edge Learning
tags:
  - []
---

"Local2Global: A distributed approach for scaling representation learning on graphss"이란 논문에 대한 리뷰입니다.

원문은 [링크](https://arxiv.org/abs/2201.04729)에서 확인할 수 있습니다.

# 3. local2global Algorithm

"to embed different parts of a graph independently by splitting the graph into **overlapping patches** and then **stitching** the patch node embeddings together to obtain a single global node embedding for each node. 
"

Stitching : estimating the orthogonal transformations(i.e., rotations and reflection) and translations, which aligns based on the overlapping nodes.

기본적으로 independently and distributed learning environment에선 scaling, reflection, rotation, translation and nois가 embedding를 다르게 만든다고 생각하여 이러한 transformation을 학습하는 것이고, 그 방법으로 한 패치 그룹(device)에서 **relative transformation**을 이용하는 방법을 제안한다.

Step 1 : scale & orthogonal transformation(rotation & reflection)을 학습<br>
- 이 때 사용하는 방법으로 **eigenvector synchronization method** 이용
- 데이터 사이즈가 커진다면 eigenvector 연산이 computational bottleneck이 될 가능성 존재

Step 2 : translation 학습<br>
- least square problem 이용 at the patch level


단 이 아이디어들은 graph domain에서만 적용되는듯하다.