---
title: "[Paper] A Survey on Deep Transfer Learning"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Transfer Learning
tags:
  - [survey]
---

"A Survey on Deep Transfer Learning"이란 논문에 대한 리뷰입니다.

원문은 [링크](https://link.springer.com/chapter/10.1007/978-3-030-01424-7_27)에서 확인할 수 있습니다.

# Definition
learning task is a non-linear function that reflected a deep neural network.

# Category
-	Iance-based methodology
  - Utilize instances in source domain by appropriate weight

Specific weight adjustment and select partial instances from source domain <br>
Assumption: partial can be utilized with appropriate weights<br>
 > TrAdaBoost: use AdaBoost to filter out and reweight them for classification
 > Bi-weighting Domain Adaptation (BIW): align feature spaces and assign weight

-	Mapping-based methodology
 - Mapping instances from two domains into a new data space with better similarity
  > Transfer component analysis (TCA) : MMD라는 loss를 추가하고, 이는 두 domain distribution의 차이를 줄여주는 term이다.

-	Network-based methodology
 - Reuse the partial of network pre-trained in the source domain

Network structure와 connection을 target domain의 model로 일부 붙이는 것<br>
Assumption: Feature extractor가 따로 있고 extracted feature는 versatile하다<br>
특히 network structure와 transferability간 관계가 중요한데, transfer하기에 용이한 network가 있다는 것이고, LeNet, AlexNet, VGG, ResNet 같은 것들이 이에 해당한다.<br>

-	Adversarial-based methodology
 - Use adversarial technology to find transferable features that both suitable 

Use GAN to find transferable representations that is applicable to both.<br>
Assumption: good representation is discriminative for the task but indiscriminative for domain.<br>
Domain Adaptation이 많이 보인다. 그래서 GAN이기도 한 것 같다.
  > Domain adaptation regularization loss term
  > Augmentation with few standard layers and a simple gradient reversal layer
