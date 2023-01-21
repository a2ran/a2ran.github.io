---
title: "[nlp_paper] Seq2Seq tensorflow implementation"
toc: true
use_math: true
categories:
  - paper
tags:
  - [keras, tensorflow, paper, model, implementation]
date: 2023-01-22
last_modified_at: 2023-01-22
sitemap:
  changefreq: daily
  priority: 1.0
---

"딥러닝을 이용한 자연어 처리 입문" 강의 MTL 모델 코드를 참고했습니다. https://wikidocs.net/24996

## <span style = "color : blue"> Model Implementation </span>

Seq2seq architecture can be implemented in virtual environments by using **tensorflow keras** model. <br>
I have constructed a simple **Seq2seq character-level neural machine translation** from the famous public-data ***"fra-eng.zip"*** <br> <br>
Source : https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html <br>
Download link: http://www.manythings.org/anki

```python
import pandas as pd
import tensorflow as tf
import urllib3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
```

```python
lines = pd.read_csv('fra.txt', names = ['src', 'tar', 'lic'], sep = '\t')
del lines['lic']
lines = lines.loc[:, 'src':'tar']
lines = lines[0:50000]
# <sos> -> \t, <eos> -> \n
lines.tar = lines.tar.apply(lambda x : '\t '+ x + ' \n')

print('# of all samples :',len(lines))
```

```python
lines.sample(5)
```

|**src**|tar|
|:---:|---|
|School is over now.|L'école est finie, désormais.|
|You're remarkable.	|Tu es remarquable.|
|We'll be neighbors.|Nous serons voisins.|
|You deserve a medal.|Tu mérites une médaille.|
|That's what I say.|C'est ce que je dis.|

### <span style = "color : skyblue"> Encodings & Paddings </span>

```python
# Define source sentence character set
src_voc = set()
for line in lines.src:
    for _ in line:
        src_voc.add(_)

# Define src character set size
src_voc_s = len(src_voc) + 1

# Index each characters
src_voc = sorted(list(src_voc))
src_index = dict([(v, k + 1) for k, v in enumerate(src_voc)])

# Encode sequence data into numerical vectors
enc_input = []

for line in lines.src:
    enc_line = []
    
    for _ in line:
        enc_line.append(src_index[_])
        
    enc_input.append(enc_line)

# Pad src sequences
max_src_len = max([len(line) for line in lines.src])
enc_input = pad_sequences(enc_input, maxlen = max_src_len, padding='post')
enc_input = to_categorical(enc_input)
```

```python
# Define target sentence character set
tar_voc = set()
for line in lines.tar:
    for _ in line:
        tar_voc.add(_)

# Define tar character set size        
tar_voc_s = len(tar_voc) + 1    

# Index each characters
tar_voc = sorted(list(tar_voc))
tar_index = dict([(v, k + 1) for k, v in enumerate(tar_voc)])

# encode target sentences into numerical vectors
dec_input = []

for line in lines.tar:
    enc_line = []
    for _ in line:
        enc_line.append(tar_index[_])
        
    dec_input.append(enc_line)

# encode target label sentences into numerical values
# remove <sos> token from sentences
dec_target = []

for line in lines.tar:
    timestep = 0
    enc_line = []
    for _ in line:
        if timestep > 0:
            enc_line.append(tar_index[_])
            timestep += 1
            
    dec_target.append(enc_line)

# Pad tar sequences
max_tar_len = max([len(line) for line in lines.tar])
dec_input = pad_sequences(dec_input, maxlen = max_tar_len, padding = 'post')
dec_input = to_categorical(dec_input)
dec_target = pad_sequences(dec_target, maxlen = max_tar_len, padding = 'post')
dec_target = to_categorical(dec_target)
```

### <span style = "color : skyblue"> Train Seq2seq Machine Translation Model </span>

```python
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
```

```python
# Define input sequences and LSTM layer
enc_inputs = Input(shape= (None, src_voc_s))
enc_lstm = LSTM(units = 256, return_state = True)

# output of LSTM layer
enc_outputs, state_h, state_c = enc_lstm(enc_inputs)

# context vector holds two states: hidden-state, cell-state
# Handle over these information to the decoder
context_vector = [state_h, state_c]
```

```python
# Define input sequences and LSTM layer
dec_inputs = Input(shape = (None, tar_voc_s))
dec_lstm = LSTM(units = 256, return_sequences = True, return_state = True)

# Decoder uses context vector from encoder cell as an initial state
dec_outputs, _, _= dec_lstm(dec_inputs, initial_state = context_vector)

# Use softmax to compare loss
dec_softmax = Dense(tar_voc_s, activation = 'softmax')
dec_outputs = dec_softmax(dec_outputs)

# Define checkpoint for best model
checkpoint_path = 'my_checkpoint.ckpt'

checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss',
                             verbose=1)

epochs = 50

# Define & Train model
model = Model([enc_inputs, dec_inputs], dec_outputs)
model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy")
history = model.fit(x=[enc_input, dec_input],
                    y = dec_target,
                    batch_size = 64,
                    validation_split = 0.2,
                    callbacks = [checkpoint],
                    epochs = epochs)

model.load_weights(checkpoint_path)

model.save("english-french-mtl.h5")
```

### <span style = "color : skyblue"> Evaluate Seq2seq Machine Translation Model </span>

```python
# Define Encoder
enc_model = Model(inputs = enc_inputs, outputs = context_vector)

# Define Decoder
dec_input_h = Input(shape=(256, ))
dec_input_c = Input(shape=(256, ))
dec_s_inputs = [dec_input_h, dec_input_c]

# Use t-1 cell and hidden state to predict next t cell
dec_outputs, state_h, state_c = dec_lstm(dec_inputs, initial_state = dec_s_inputs)

dec_states = [state_h, state_c]
dec_outputs = dec_softmax(dec_outputs)
dec_model = Model(inputs = [dec_inputs] + dec_s_inputs,
                  outputs = [dec_outputs] + dec_states)

# get vocab from index
index_to_src = dict((i, char) for char, i in src_index.items())
index_to_tar = dict((i, char) for char, i in tar_index.items())
```

```python
def dec_sequence(input_seq):
    states_value = enc_model.predict(input_seq)
    
    # create one-hot vector for <sos> token
    target_seq = np.zeros((1, 1, tar_voc_s))
    target_seq[0, 0, tar_index['\t']] = 1.
    
    stop_condition = False
    dec_sentence = ""
    
    # Run through iteration until stop_condition becomes TRUE
    while not stop_condition:
        # Use hidden-state + t-1 cell to predict t cell
        output_tokens, h, c = dec_model.predict([target_seq] + states_value)
        
        # tokenize predictions into word sequence
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar[sampled_token_index]

        # Append predicted word into decoded sentence
        dec_sentence += sampled_char

        # If dec-sentence reaches <eos>, break the iteration.<eos>
        if (sampled_char == '\n' or
            len(dec_sentence) > max_tar_len):
            stop_condition = True

    # Save t cell
    target_seq = np.zeros((1, 1, tar_voc_s))
    target_seq[0, 0, sampled_token_index] = 1.

    # Save hidden state
    states_value = [h, c]
    
    return dec_sentence
```

```python
for _ in [3, 50]:
    seq = enc_input[_: _ + 1]
    dec_sentence = dec_sequence(seq)
    print('-' * 35)
    print('input sentence: ', lines.src[_])
    # print without \t and \n
    print('answer sentence: ', lines.tar[_][2:len(lines.tar[_]) - 1])
    # print without \n
    print('translated sentence: ', dec_sentence[1: len(dec_sentence) - 1])
```

-----------------------------------
입력 문장: Hi. <br>
정답 문장: Salut ! <br>
번역 문장: Salut. <br>
-----------------------------------
입력 문장: I see. <br>
정답 문장: Aha. <br>
번역 문장: Je change. <br>
-----------------------------------
