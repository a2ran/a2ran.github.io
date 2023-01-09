---
title: "[nlp_models] Huggingface Transformer Model Review : ko-sroberta-multitask"
toc: true
use_math: true
categories:
  - model
tags:
  - [nlp, transformer, model]
date: 2023-01-09
last_modified_at: 2023-01-09
sitemap:
  changefreq: daily
  priority: 1.0
---

kr sentence transformer model review and implementation

# <span style = "color : skyblue"> Installation </span>

ko-sroberta-multitask model is a korean sentence feature-extraction model trained by RoBERTa model. <br>
It can map korean sentences and paragraphs into 768 dimensional dense vectore space. <br>
Korean transformer models can be installled from Huggingface via pip install library

```python
!pip install sentence_transformers

from sentence_transformers import SentenceTransformer
```

requirements are listed below

```python
pip install python >= 3.6.0
pip install pytorch >= 1.6.0
pip install transformers >= 4.6.0

## <span style = "color : blue">Model Architecture</span>

korean sentence encoding model 'ko-sroberta-multitask' model is trained from following parameters

```python
{
    "epochs": 5,
    "evaluation_steps": 1000,
    "evaluator": "sentence_transformers.evaluation.EmbeddingSimilarityEvaluator.EmbeddingSimilarityEvaluator",
    "max_grad_norm": 1,
    "optimizer_class": "<class 'transformers.optimization.AdamW'>",
    "optimizer_params": {
        "lr": 2e-05
    },
    "scheduler": "WarmupLinear",
    "steps_per_epoch": null,
    "warmup_steps": 360,
    "weight_decay": 0.01
}
```

## <span style = "color : blue">Model Evaluation</span>

These are the performance score ko-sroberta model acquired from training KorSTS, KorNli Datasets

```python
"Cosine Pearson": 84.77
"Cosine Spearman": 85.60
"Euclidean Pearson": 83.71
"Euclidean Spearman": 84.40
"Manhattan Pearson": 83.70
"Manhattan Spearman": 84.38
"Dot Pearson": 82.42
"Dot Spearman": 82.33
```

# <span style = "color : skyblue"> Usage & Implementation </span>

One can simply install pretrained model to encode sentences into pretrained [1,768] tensor.

```python
# Requisites
from transformers import AutoTokenizer, AutoModel
import torch

# Load model from HuggingFace.co
tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sroberta-multitask')
model = AutoModel.from_pretrained('jhgan/ko-sroberta-multitask')

# Perform mean pooling to down shape while preserving important features
def mean_pooling(output, attn):
    token = output[0] #FIrst element of output contains all token embeddings
    input_expanded = attn.unsqueeze(-1).expand(token.size()).float()
    return torch.sum(token * input_expanded, 1) / torch.clamp(input_expanded.sum(1), min = 1e-9)

sentence = ['안녕하세요. 이것은 한국어 문장입니다.']

# Tokenize
encoder = tokenizer(sentence, padding = True, truncation = True, return_tensors = 'pt')
                    
# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoder)         
                    
# Perform mean pooling
sentence_embeddings = mean_pooling(model_output, encoder['attention_mask'])
               
# Result                    
print(sentence_embeddings.shape)
# output = torch.Size([1, 768])
```
## <span style = "color : blue">Train from private dataset</span>

Or if we want to train model from our private dataset, we can use SentenceTransformer package to transform datas into vectors. <br>
Here is an example of encoding all_is_wellness psychological counseling dataset from AI-hub.com.

```python
from sentence_transformers import SentenceTransformer

# Update model from Huggingface.co
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

# Get dataset from local environement
def get_dataset():
    df = pd.read_csv('all_wellness.csv')
    return df

model = cached_model()
df = get_dataset()
df = df.head()

# encode question column into (1, 768) vectors
df['embedding'] = df.apply(lambda row: model.encode(row.question), axis = 1)
```
Result:

|**question**|response|embedding|
|:---:|---|---|
|제 감정이 이상해진 것 같아요. 남편만 보면 화가 치밀어 오르고 감정 조절이 안되요.|감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.|[-0.4806059, -0.2948694, 0.4379001, -0.6401378...|

## <span style = "color : blue">Implementation</span>

embedded word vecotrs have a variety of usages. <br>
One such example is constructing a chatbot environment. <br>
A chatbot model can train data from embedded word vectors, and make a plausible response based on cosine_similarity equation. <br>
A simple question & response example is visualized below. <br>
If an user can attain enough question and response datas, one can simply build a chatbot using this code and distribute it via streamlit.io

```python
# Requisites
from sklearn.metrics.pairwise import cosine_similarity

# Define input sentence
user_input = str(input("input: "))

# Encode sentence from pretrained model
embedding = model.encode(user_input)

# Scrutinize similar vector using cosine similarity equation
df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())

# Define answer sentence
answer = df.loc[df['distance'].idxmax()]

# Print out response
print("response:", answer['response'])
```

Results are shown below.

```python
input:더 이상 내 감정을 내가 컨트롤 못 하겠어.
response: 감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.
```
