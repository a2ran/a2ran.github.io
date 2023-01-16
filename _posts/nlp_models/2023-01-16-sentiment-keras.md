---
title: "[nlp_models] Basic sequential model based on Tenserflow Keras library"
toc: true
use_math: true
categories:
  - model
tags:
  - [keras, tensorflow, model]
date: 2023-01-16
last_modified_at: 2023-01-16
sitemap:
  changefreq: daily
  priority: 1.0
---

How to: analyze 200,000 sized movie sentiment data through Bidirectional LSTM with ease.

## <span style = "color : skyblue"> Retrieve Data from githubcontents </span>

"Naver Moive Review" is a 200,000 sized public korean text dataset that is commonly used in academic sentiment analysis purposes. <br>
It can be retreieved directly from github.
Installation code is listed below.

```python
import numpy as np
import pandas as pd
import urllib.request

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename= "ratings.txt")
train = pd.read_table('ratings.txt')
```

## <span style = "color : skyblue"> Data Overview </span>

The dataset encodes $1$ for positive reviews and $0$ for negative reviews. <br>
Check out whether there are any missing values and remove them if have any.

```python
# 1 for positive review
# 0 for negative review

print(train.shape)
print('-'*10)
print(train.isnull().sum())
train.dropna(axis = 0, inplace = True)
print('-'*10)
train.head()
```
<br>

> (200000, 3) <br>
> ---------- <br>
> id          0 <br>
> document    0 <br>
> label       0 <br>
> dtype: int64 <br>
> ---------- <br>

|**id**|document|label|
|:---:|---|---|
|8112052|어릴때보고 지금다시봐도 재밌어요ㅋㅋ|1|
|8132799|디자인을 배우는 학생으로, 외국디자이너와 그들이 일군 전통을 통해 발전해가는 문화산...|1|
|4655635|폴리스스토리 시리즈는 1부터 뉴까지 버릴께 하나도 없음.. 최고.|1|

## <span style = "color : skyblue"> Preprocess Data </span>

Atypical data requires preprocessing to boost accuracy and prediction results. <br>
Simply remove non-korean words using **regular expressions**.

```python
train['document'] = train['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
```

## <span style = "color : skyblue"> Split Train & Validation Data </span>

Split dataset into trainable train_set and validation_set used for validation. <br>
Scikit-learn library **train_test_split** is a commonly used option.

```python
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(train['document'], train['label'], test_size = 0.2, stratify = train['label'], random_state = 29)
```

## <span style = "color : skyblue"> Define Variables </span>

It is better to define variables used in the sequence firsthand. <br>
The variables used in the model is listed below.

```python
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
```

## <span style = "color : skyblue"> Tokenize & Pad Sentences </span>

In order for Keras model to understand and process data, one needs to encode natural language sentences into numerical vectors. <br>
Tensorflow provides **Tokenizer** and **pad_sequences** library to facilitate this processs. <br>
<br>
**Tensorflow Tokenizer** tokenizes words based on word frequencies. <br>
**Tensorflow pad_sequences** pads tokenized vectors into unified lengthed vectors.
Code used in the model is listed below.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define & fit tokenizer
tokenizer = Tokenizer(num_words = vocab_size, oov_token = '')
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index

# Tokenize sentences
train_sequences = tokenizer.texts_to_sequences(x_train)
val_sequences = tokenizer.texts_to_sequences(x_val)

# Pad tokenized sentences
train_padded = pad_sequences(train_sequences, maxlen = max_length, truncating = trunc_type, 
                               padding = padding_type)
val_padded = pad_sequences(val_sequences, maxlen = max_length, padding = padding_type, 
                                    truncating = trunc_type)
```                                    
                                    
## <span style = "color : skyblue"> Define Model </span>

**Tensorflow Sequential** library facilitates pipeline building process. <br>
**Embedding** layer maps discrete symbols to continuous vectors to use as an input layer of the network. <br>
**Dropout** potential overfitting data. <br>
**Bidirectional-LSTM** processes input sequences in both forward and backward directions. It is useful when the context of the input sequence is important for the task at hand. <br>
Since the output label is $0$ and $1$, dense layer into 1-dimensional sized sigmoid function. <br>
Finally, train the model given the tensorflow pipeline
<br>
<br>

```python
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten, Dropout
from tensorflow.keras.models import Sequential

model = Sequential([Embedding(vocab_size, embedding_dim, input_length=max_length),
                    Dropout(0.5),
                    Bidirectional(LSTM(64, return_sequences=True)),
                    Bidirectional(LSTM(64)),
                    Dense(32, activation='relu'),
                    Dense(16, activation='relu'),
                    Dense(1, activation='sigmoid')
                   ])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
```

## <span style = "color : skyblue"> ModelCheckpoint and Load Optimized Weights  </span>

Tensorflow provides function to save and load weights from best epoch output. <br>
Recommended epoches for the model is over ten, but I have used three since the GPU environment was not available ATM. <br>
The checkpoint code and loading best weight based on validation loss is listed below.

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = 'my_checkpoint.ckpt'

checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss',
                             verbose=1)
  
epochs = 3
  
history = model.fit(train_padded, y_train, 
                    validation_data=(val_padded, y_val),
                    callbacks=[checkpoint],
                    epochs=epochs)
  
model.load_weights(checkpoint_path)

model.save("naver-movie-analysis.h5")
```

<img src = '/assets/images/nlp_study/model/keras_1.png' width = '900'>

## <span style = "color : skyblue"> Evaluation  </span>

Evaluate validation loss and accuracy using model.evaluate function. <br>
The optimized output from the model is :

> val_loss = 0.4145 <br>
> val_acc = 0.7900 <br>

The result can be further improved by optimizing model pipeline and increasing number of epoches.

```python
model.evaluate(val_padded, y_val)
```

> [0.4145113527774811, 0.7900197505950928]
