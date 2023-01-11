---
title: "[basics] Stanford CS224n: Lecture 1 - Intro & Word Vectors - Lecture Review"
toc: true
use_math: true
categories:
  - study
tags:
  - [study, NLP. word2vec, lecture]
date: 2023-01-11
last_modified_at: 2023-01-11
sitemap:
  changefreq: daily
  priority: 1.0
---

Week 1 task of Stanford CS244n: Natural Language Processing with Deep Learning


# <span style = "color: blue"> Lecture (강의 내용) </span>

## <span style = "color : skyblue"> 1. Introduction (~ 16:01) </span>

16:01분까지는 전반적인 개요에 대한 설명이 주를 이루고 있습니다. 담당교수가 NLP에 대해 가지고 있는 생각을 중점으로 강의를 들으시면 되겠습니다.

* Key question for artifical intelligence and human-computer interaction is how to get computers to be able to understand the information conveyed in human languages

> Artificial Intelligence, 즉 인공지능과 인간-컴퓨터 상호작용에 있어 중요한 요소 중 하나는 컴퓨터가 어떻게 인간 언어에 담긴 맥락을 이해할 수 있을까에 대한 질문입니다.

<br>

### **gpt-3 모델이 중요하다고 교수가 생각하는 이유:**

* gpt-3 model is the first step on *universal models*. It is a trained up one extremely large model on every knowledge of human world. So people no longer need specific models. Just one single model that understands anything.

> 교수는 GPT-3 모델이 *유니버셜 모델*로 가기 위한 첫번째 단계라고 생각합니다.
> 유니버셜 모델은 인간 사회의 모든 지식을 담은 매우 방대한 모델로서, 상황에 따라 각자의 모델을 사용하는 것이 아닌 모든 것을 이해하는 단 하나의 모델만을 사용하게 만드는 모델입니다.

<br>

<img src = '/assets/images/nlp_study/week1/week1_1.png' width = '700'>

### **gpt-3 모델 작동 예시**

* gpt-3 can predict following words, but just predicting one word at a time to complete text.
> GPT-3 모델은 따라오는 단어를 하나씩 예측하여 텍스트를 완성할 수 있습니다. <br> 예시 : 일론 머스크 트위터
* if given couple of examples, gpt-3 can follow ideas and patterns.
> GPT-3 모델에 예시들을 주면, GPT-3 모델은 그 예시의 패턴과 아이디어를 따라할 수 있습니다. <br> 예시: 문장을 SQL 문법의 문장으로 바꿀 때
