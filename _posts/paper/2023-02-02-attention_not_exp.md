---
title: "[nlp_paper] Attention is not Explanation Paper Review"
toc: true
use_math: true
categories:
  - paper
tags:
  - [paper, review, model]
date: 2023-02-02
last_modified_at: 2023-02-02
sitemap:
  changefreq: daily
  priority: 1.0
---

- DSL 논문 스터디 6기 손예진님이 발제하신 내용을 기반으로 작성했습니다.

## Attention

<img src = '/assets/images/paper/xai_attention/1.png'>

The **Attention** model works by a specific attention mechanism ***"Scaled Dot Product Attention"*** <br>
<br>
$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt(d_k)})$ <br>
<br>

Attention mechanism splits the input into three categories: <br>

> ($Q$) for *query* (hidden-state at t decoder cell)<br>
> ($K$) for *key*  (hidden-state at every encoder cell $k$)<br>
> ($V$) for *value*  (hidden-state at every encoder cell $v$)

<br>

By calculating the **Dot product** of query $Q$ and key $K$, and scaling it by the **square root** of key $k$'s dimension, <br>
The attention mechanism produce a **score** for each key value pair that indicates the importance of value $v$.

<br>

By computing scores for each value $v$, the **Attention** model is able to decide a particular important part of the **src sentences** it should attend to when making every predictions. 

## Is Attention Explainable?

Attention is not Explanation (Jain and Wallace; 2019, NAACL)

<img src = '/assets/images/paper/xai_attention/2.png'>

The **Attention model** has been known to have ***Explainability***<br>
since it can sort out **important input tokens** that affect the result of classification.

For instance, by highlighting important tokens in the source sentences,<br> one can find out if the model is designed as it was meant to be, <br> and use it to **explain** the result of the model.

<img src = '/assets/images/paper/xai_attention/3.png'>

However, there are several reported cases of attention model **not properly** repersenting the explainability of tokens.

*Attention Calibration for Transformer in Neural Machine Translation (Lu et al., ACL; 2021)* found out that some outputs (e.x. "Deaths") were tokenized from negligible tokens such as **[EOS]**.
  
Also, *What does BERT look at? An Analysis of BERT's Attention (Clark et al., BlackboxNLP; 2019* claims that most of attention results are tokenized from tokens such as [CLS], [SEP], Periods, and Commas.

So, there are several **counterexamples** to use Attention to explain models.

## Paper Review

***Main Opinion***: Attention is not a **faithful explanation** to a model.

Attention model should satisfy following constraints to become explainable.

> 1. Attention weights should be similar to other **explainable methods** such as **Feature Importance.** <br>
> 2. There should be only one attention for a single output.

### Experiment 1-1

<img src = '/assets/images/paper/xai_attention/4.png'>

**Experiment 1-1** explains correlation between **Attention Weights** and **Feature Importance** (Gradient/Leave One Out)

**Gradient** calculates how change in a **particular token** affects the predicted value.

$\tau_g$ = correlation between Gradient and Attention.

**Leave One Out (LOO)** calculates how excluding a **particular token** affects the overall performance of the model.

$\tau_{loo}$ = correlation between the output and attention

### Result

<img src = '/assets/images/paper/xai_attention/5.png'>

The **correlation coefficient** between Attention-Importance is lower than the **Average** value.

$\therefore$ Attention Weights have **low correlation** with Feature Importance.

### Experiment 1-2

**Experiment 1-2** explores the correlations between feature importance and attention weights.

<img src = '/assets/images/paper/xai_attention/6.png'>

### Result

The correlation between LOO & Gradient is **significantly remarkable** than the correlation between Attention & Importance (0.2 ~ 0.4)

### Experiment 2

**Experiment 2** explores if it is possible to create another **attention** that yields same output with original attention model.

<img src = '/assets/images/paper/xai_attention/7.png'>

**Total Variance Distance (TVD)** calculates difference between outputs

**Jensen Shannon Divergence (JSD)** calculates difference between two distributions.

**Counterfactual Attention** is an attention model that yields same output with original attention model.

### Experiment 2-1

**Permutated Attention** randomly permutates attention to calculate a **total variance distance** between original attention output and the permutated attention output.

**Experiment 2-1** searches if the counterfactual attention can be built from an original model with a miniscule difference in output.

<img src = '/assets/images/paper/xai_attention/8.png'>

### Result

<img src = '/assets/images/paper/xai_attention/9.png'>

Above graphs are the median difference between the $y$ from the original attention and $y$ from permutation attention.

It is shown that there is no difference between permutated output and the original output.

## Conclusion

**Experiment 1 :** Attention Weights is not related to Feature Importance. <br>
However the correlation between Feature Importance is significant.

**Experiment 2 :** There can be multiple attention for a single output. <br>
The difference between Adversarial attention with original attention is miniscule.

**As a Result**, Attention is not Explaination.

Source : https://arxiv.org/pdf/1902.10186.pdf<br>
DSBA seminar material : http://dsba.korea.ac.kr/seminar/
