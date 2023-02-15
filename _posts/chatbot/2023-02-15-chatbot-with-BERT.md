---
title: "phychotherapy chatbot with BERT"
toc: true
use_math: true
categories:
  - chatbot
tags:
  - [projects, tensorflow, nlp, transformer, BERT]
date: 2023-02-15
last_modified_at: 2023-02-15
sitemap:
  changefreq: daily
  priority: 1.0
---

웰니스 대화 스크립트 데이터셋(AI Hub, 2019)을 사용한 질의응답용 챗봇 프로그램입니다.

## Introduction

<img src = '/assets/images/projects/chatbot/bert/1.png'>

## ko-sROBERTa-Multitask

<img src = '/assets/images/projects/chatbot/bert/4.png'>

<img src = '/assets/images/projects/chatbot/bert/5.png'>

## 코드 예시

```python
import urllib.request
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('jhgan/ko-sroberta-multitask')

df['embedding'] = df.apply(lambda row: model.encode(row.question), axis = 1)
```

$cosine similarity = \large \frac{A \cdot B}{\left | A \right | \left | B \right |}$

```python
def cosine_similarity(A, B):
    return dot(A, B)/(norm(A) * norm(B))
    
def return_response(input):
    embedding = model.encode(input)
    df['score'] = df.apply(lambda x: cosine_similarity(x['embedding'], embedding), axis = 1)
    return df.loc[df['score'].idxmax()]['response']
```

## Result

<img src = '/assets/images/projects/chatbot/bert/3.png'>

### 코드 예시

```python
q = '몸이 아파서 제대로 생활하지 못하겠어요.'
print(f'Q : ', q)
rsp = return_response(q)
print(f'A : ', rsp)
print('-' * 30)
cnt = len(df[df['response'] == rsp])
print(f'데이터 내 같은 답변의 개수 : {cnt}개')
ind = df.index[df['score'].idxmax()]
print(f'답변을 가져온 데이터 index 번호 : {ind}')
```

> Q :  몸이 아파서 제대로 생활하지 못하겠어요. <br>
> A :  내담자분이 제대로 휴식을 취하지 못하고 계신 거 같아 걱정스러워요. <br>
> ------------------------------ <br>
> 데이터 내 같은 답변의 개수 : 64개 <br>
> 답변을 가져온 데이터 index 번호 : 6987
