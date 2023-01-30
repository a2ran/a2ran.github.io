---
title: "[study] Stanford CS224N: Lecture 4 - Synthetic Structure and Dependency Parsing"
toc: true
use_math: true
categories:
  - study
tags:
  - [study, NLP, Parsing, lecture]
date: 2023-01-30
last_modified_at: 2023-01-30
sitemap:
  changefreq: daily
  priority: 1.0
---

Week 4 task of Stanford CS244n: Natural Language Processing with Deep Learning 

# <span style = "color: blue"> Lecture </span>

## Constituency Structure

<img src = '/assets/images/nlp_study/week4/1.jpg'>

**Constituency Structure** parses a sentence into nested constituents <br>
to figure out the **structure of the sentence.**

For instance,

"*John hit the ball*"

1. Identify the noun (N) from sentence (S) = "John"
2. Identify the verb (V) from verb phrase (VP) = "hit"
3. Identify the detrient (D) from Noun phrase (NP) = "the"
4. Label each phrases and represent the relationship using a tree diagram

**Constituency Structure** has its usage on **understanding sentence structures.**

## Dependency Structure

<img src = '/assets/images/nlp_study/week4/2.png'>

**Dependency Structure** has its usage on **understanding relationship between vocabularies.**

1. The **Arrow** heads from "head" to "dependent". <br>
2. The **Label** above the arrow indicates the **dependency** between vocabularies.
3. To make all words dependent, add one or more fake **ROOTs.**
4. Arrows do not cycle (if A -> B, then B !-> A)

### Transition-based Dependency Parsing

<img src = '/assets/images/nlp_study/week4/3.png'>

**Transition-based parser** determines the **dependency** of two words in a **sequential order.**

$c = (\sigma, \beta, A)$ where $\sigma$ = STACK, $\beta$ = BUFFER, $A$ = Set of Arcs

All decisions are made from $f(c)$ which inputs State (c) in a BUFFER.

1. ROOT | I ate fish | X

2. ROOT I | ate fish | X

Decision Process : Shift

3. ROOT I ate | fish | X

Decision Process : Shift -> Shift

4. ROOT ate | fish | (ate, nsubj, I)

Decision Process : Shift -> Shift -> Left-Arc (nsubj)

5. ROOT ate fish | X | (ate, nsubj, I)

Decision Process : Shift -> Shift -> Left-Arc (nsubj) -> Shift

6. ROOT ate | X | (ate, nsubj, I) (ate, dobj, fish)

Decision Process : Shift -> Shift -> Left-Arc (nsubj) -> Shift -> Right-Arc (dobj)

7. ROOT | X | (ate, nsbj, I) (ate, dobj, fish) (ROOT, root, ate)

**Decision Process : Shift -> Shift -> Left-Arc (nsubj) -> Shift -> Right-Arc (dobj) -> Right-Arc (root)**
