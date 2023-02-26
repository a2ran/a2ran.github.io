---
title: "Sequence Classification of IMDB reviews with LSTM"
toc: true
use_math: true
categories:
  - app
tags:
  - [Deep Learning, RNN, classification, projects]
date: 2023-02-26
last_modified_at: 2023-02-26
sitemap:
  changefreq: daily
  priority: 1.0
---

Classifying IMDB review sentiments with Bidirectional-LSTM architecture

## Retrieve Data

```python
import urllib.request

url = "https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv"
urllib.request.urlretrieve(url, filename = "IMDb_Reviews.csv")

df = pd.read_csv('IMDb_Reviews.csv', encoding='utf-8')
```

## Preprocessings

### Train_Test_Split

```python
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df, test_size = 0.20, random_state = 77)
print('훈련용 리뷰의 개수 :', len(train_data))
print('테스트용 리뷰의 개수 :', len(test_data))
```
> 훈련용 리뷰의 개수 : 40000 <br>
> 테스트용 리뷰의 개수 : 10000

### Tokenize & Pad Sequences

```python
import nltk

nltk.download('punkt')
nltk.download('treebank')
```

```python
from nltk.tokenize import word_tokenize

train_data['review'] = train_data['review'].apply(word_tokenize)
test_data['review'] = test_data['review'].apply(word_tokenize)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['review'].values)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
```

## Define LSTM Model

```python
import re
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(hidden_units)))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size = 256, validation_split = 0.2)
```

## Result

```python
loaded_model = load_model('best_model.h5')

print("테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
```

> 테스트 정확도: 0.8911

### Prediction

```python
def sentiment_predict(new_sentence):
    new_sentence = word_tokenize(new_sentence) # 토큰화
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
    score = float(loaded_model.predict(pad_new)) # 예측
    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))
```

> sentiment_predict('The greatest of the 20th century.')

> 86.69% 확률로 긍정 리뷰입니다.
