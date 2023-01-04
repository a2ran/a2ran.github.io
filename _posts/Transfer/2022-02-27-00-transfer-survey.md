---
title: "[Paper] A survey of transfer learnings"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Transfer Learning
tags:
  - [survey]
---

"A survey of transfer learning"이란 논문에 대한 리뷰입니다.

원문은 [링크](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-016-0043-6)에서 확인할 수 있습니다.

# Terminology
Feature space Χ / Label space Y
Predictive Function f(∙)=P(Y│X)
Domain D={Χ,P(Χ)}
Task Τ={Y,f(∙)}

| |Source|Target|
|:---:|:---:|:---:|
|Domain|$X_{S},P(X_{S})$| $X_{T},P(X_{T})$|
|Domain data|$D_S={(x_{S1},y_{S1} ),…,}	$|D_T={(x_{T1},y_{T1}),…,}|
|Task|$Y_{S},f_{S}(∙)$|$Y_{T},f_{T}(∙)$|

# Definition
improving target predictive function using source domain data & target domain data.
 - X, P(X)가 달라서 생기는 문제로 귀결된다.
 - Domain Adaptation이라는 것과 혼용되지만 이건 source domain을 target domain과 비슷하게 만드는 방식이다.

# Taxonomy
|Feature Space	Homogeneous| Heterogeneous|
|:---:|:---:|
|Predictive Function|Mismatch in the conditional prob|
|Label Space|Mismatch in the class space|
|P(Y)	| Caused by labelled and unlabelled|

# Bias
|||
|:---:|:---:|
|Frequency feature bias	 |P(X_S )≠P(X_T )|
|Context Feature bias	 |$P(Y_{S}|X_{S})≠P(Y_{T}|X_{T}})$|


# Generic Solution	
|||
|:---:|:---:|
|Homogeneous	| Correct marginal and/or conditional|
|Heterogeneous	|(Same domain distribution) Align Input space|
||	(Different domain distribution) domain adaptation|


# General Strategy
|||
|:---:|:---:|
|Information transfer|(Through Instances) reweight source domain to correct marginal <br>->	Conditional is the same|	
|Information transfer|(Through features)	(Asymmetric transformation) Transform features through reweighting|
|Information transfer| (Symmetric transformation)Find common latent feature space|
|Information transfer|(Through Parameter, Ensemble learning) multiple source learners |
|Information transfer|Transfer based on some defined relationship(least)|

Generic solution + General strategy  Methodology 

