---
title: "transformer-chatbot"
toc: true
use_math: true
categories:
  - chatbot
tags:
  - [projects, tensorflow, nlp, transformer]
date: 2023-02-11
last_modified_at: 2023-02-11
sitemap:
  changefreq: daily
  priority: 1.0
---

Tensorflow tutorial을 참고하여 만든 transformer모델 기반 한국어 심리상담 챗봇 미니 프로젝트입니다.

ipynb 파일 폴더 : [https://github.com/a2ran/kor-eng-translator](https://github.com/a2ran/transformer-chatbot)<br>
참고문헌 : [https://www.tensorflow.org/text/tutorials/transformer](https://www.tensorflow.org/text/tutorials/transformer)<br>
사용 데이터셋: 웰니스 대화 스크립트 데이터셋(AI Hub, 2019)

## 한국어 문장 형태소 분석

<img src = '/assets/images/projects/chatbot/1.png'>

```python
# 한국어 형태소 분석기 Mecab을 정의합니다.

from eunjeon import Mecab
import re

m = Mecab()

def reg_kor(sentence):
    
    # 한국어, 숫자, 지정한 특수문자를 제외한 html 언어를 제거합니다.
    sentence = re.sub('[^0-9가-힣!?,.()]', ' ', sentence)
    
    # 문장 양옆의 빈칸을 제거하고, 특수문자를 한 칸씩 띄어씁니다.
    sentence = re.sub(r"([,.?!])", r" \1", sentence.strip())
    
    # 두 칸 이상 벌어진 문자가 있으면 한 칸으로 통일해줍니다.
    sentence = re.sub(r'[" "]+', " ", sentence)
    
    # 한국어 형태소 분석기 mecab으로 형태소 분석을 진행합니다.
    sentence = " ".join(m.morphs(sentence))
    
    return sentence
```

<img src = '/assets/images/projects/chatbot/20.png'>

## Input & Output Embeddings

<img src = '/assets/images/projects/chatbot/2.png'>

### 데이터 임베딩 + zero-padding 결과

```python
zp = src_text_processor(src_raw[:30])
plt.figure(figsize = (10, 6))

plt.subplot(1, 2, 1)
plt.pcolormesh(zp.to_tensor(), cmap = 'nipy_spectral')
plt.title('토큰 벡터 배치')

plt.subplot(1, 2, 2)
plt.pcolormesh(zp.to_tensor() != 0, cmap = 'nipy_spectral')
plt.title('zero-padding 결과')
```

<img src = '/assets/images/projects/chatbot/3.png'>

## Positional Encoding

<img src = 'https://www.tensorflow.org/images/tutorials/transformer/PositionalEmbedding.png'>

사진출처: [https://www.tensorflow.org/text/tutorials/transformer](https://www.tensorflow.org/text/tutorials/transformer)

<img src = '/assets/images/projects/chatbot/4.png'>

## Base Attention Layer

<img src = 'https://www.tensorflow.org/images/tutorials/transformer/BaseAttention.png' width = '400'>


<img src = '/assets/images/projects/chatbot/5.png'>

## Cross Attention Layer

<img src = 'https://www.tensorflow.org/images/tutorials/transformer/CrossAttention.png' width = '400'>


<img src = '/assets/images/projects/chatbot/6.png'>

## Self-Attention Layer

<img src = 'https://www.tensorflow.org/images/tutorials/transformer/CausalSelfAttention.png' width = '400'>


<img src = '/assets/images/projects/chatbot/7.png'>

## Feed Forward Layer

<img src = 'https://www.tensorflow.org/images/tutorials/transformer/FeedForward.png' width = '400'>


<img src = '/assets/images/projects/chatbot/8.png'>

## Encoder Layer

<img src = 'https://www.tensorflow.org/images/tutorials/transformer/EncoderLayer.png' width = '450'>

<img src = '/assets/images/projects/chatbot/9.png'>

## Decoder Layer

<img src = 'https://www.tensorflow.org/images/tutorials/transformer/DecoderLayer.png' width = '450'>

<img src = '/assets/images/projects/chatbot/10.png'>

## Transformer

<img src = 'https://www.tensorflow.org/images/tutorials/transformer/transformer.png' width = '400' align = 'left'>

<img src = '/assets/images/projects/chatbot/11.png'>

## Train model

<img src = '/assets/images/projects/chatbot/12.png'>

# Attention Plots

<img src = '/assets/images/projects/chatbot/13.png'>

<img src = '/assets/images/projects/chatbot/14.png'>

## Predict new sentences

<img src = '/assets/images/projects/chatbot/15.png'>
