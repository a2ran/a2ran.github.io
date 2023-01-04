---
title: "[Paper] Communication Efficient Learning"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Edge Learning
tags:
  - [Distributed, Federated]
---

"Communication Efficient Learning"이란 논문에 대한 리뷰입니다.

원문은 [링크](http://proceedings.mlr.press/v54/mcmahan17a.html)에서 확인할 수 있습니다.

Communication-Efficient Learning of Deep Networks from Decentralized Data <br>

Client(Node) – Server(Aggregator)

# Contribution
1.	Identification of training on decentralized data from mobile device
2.	Algorithm
3.	FederatedAveraging(FedAvg) Algorithm with SGD

# Federated Optimization Issues<br>
-	Non-IID & Unbalanced
-	Massively distributed: client > average num of example per client
-	Limited communication: offline or slow & expensive connections
-	(Practical) data changes
-	(Practical) client availability with the local data distribution
-	Communication costs dominate

K: num of clients<br>
C: fraction of clients data (when c=1, global batch size, full-batch)<br>
Non-convex Obejectives

# Purpose: Decrease the number of rounds of communication
1.	Increasing Parallelism: more clients working
2.	Increasing computation on each client
3.	One-shot averaging
4.	FederatedAveraging Algorithm 

# Large batch synchronous SGD
The amount of computation is controlled by <br>
-	C: fraction
-	E: Number of training passes
-	B: the local minibatch size
FederatedSGD(FedSGD) : E=1 <br>
FedAveraging(FedAvg) : what we concerns

# RESULT
## Model Architecture
Data : MNIST digit recognition <br>
Model 1) MNIST 2NN <br>
Multilayer-perceptron with 2 hidden with 200 unit using ReLu(total 199,210 params)<br>
Model 2) CNN <br>
5*5 conv layer(32 – 64 channel with 2*2 max pooling) & fully connected with 512 units and ReLu(1,663,370)<br>

## Data Distribution
1.	IID: 100 clients with 600 ex
2.	Non IID: 200 shards of size 300, 100 client with 2 shards
3.	LSTM
4.	Language Model

## Parallelism
With B = , small advantage in increasing the client fraction <br>
Smaller B, significant improvement, especially in the non=IID

Computation per client
For fixed C=0.1,
More computation per client(decreasing B, increasing E or both),
More local SGD updates per round, resulting in decreasing in communication costs.
Expected number of updates per client: u=EnkB*E=nEKB
Increasing u by varying E & B is effective
As long as B is large enough for parallelism, no cost in computation time for lowering it, 그래서 parameter to be tuned at first
More computation per client decreases the number of rounds 
FedAvg is better than the benchmark(FedSGD) 
