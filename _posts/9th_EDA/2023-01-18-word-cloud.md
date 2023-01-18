---
title: "[projects] EDA_wordcloud"
toc: true
use_math: true
categories:
  - 9th_EDA
tags:
  - [projects, EDA, Visualization]
date: 2023-01-18
last_modified_at: 2023-01-18
sitemap:
  changefreq: daily
  priority: 1.0
---

Open crawled data from naver blog reviews and represent them as wordcloud images.

## <span style = "color : blue"> Prerequisites </span>

```python
import re
import konlpy
import pandas as pd

# Open file
with open(r'./naver_blog_review_1', encoding='utf-8') as f:
    text = f.readlines()
```

## <span style = "color : blue"> Preprocess text </span>

```python
text_strip = list([i.strip() for i in text if i != '\n'])
text_join = ' '.join(text_strip)

filtered_content = re.sub(r'[^\d\s\w]', ' ', text_join)
```

## <span style = "color : blue"> Tokenize Korean Vocabs </span>

```python
komoran = konlpy.tag.Komoran()
komoran_pos = komoran.pos(filtered_content)

komoran.morphs(filtered_content)

komoran_nouns = komoran.nouns(filtered_content)

stop_words = []

def tokenizer(text):
    nouns = komoran.nouns(text)
    go_words = [noun for noun in nouns if noun not in stop_words]
    return [go_word for go_word in go_words if len(go_word)>1]
    
filtered_double = tokenizer(filtered_content)
```

## <span style = "color : blue"> Count the most Frequent vocabs </span>

```python
from collections import Counter
c = Counter(filtered_double)
frequent = c.most_common(10)
```

## <span style = "color : blue"> Wordcloud </span>

```python
# Wordcloud settings
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from os import path

FONT_PATH = "C:/windows/fonts/malgun.ttf"
```

### <span style = "color : skyblue"> Wordcloud -> "Hangang", "fine dust", "park" </span>

```python
# Wordcloud -> "Hangang", "fine dust", "park"
from wordcloud import ImageColorGenerator

img = plt.imread("blossom.jpg")

wordcloud1 = WordCloud(
        font_path=FONT_PATH,
        background_color = "black",
        random_state = 1,
        color_func = ImageColorGenerator(img),        
        mask = img
)
wordcloud1.generate_from_frequencies(c)
wordcloud1.to_image()
```

<img src = '/assets/images/projects/result_1.png' width = 500> <br>

### <span style = "color : skyblue"> Wordcloud -> "fine dust" </span>

```python
# Wordcloud -> "fine dust"
from wordcloud import ImageColorGenerator
import numpy as np

col=['#6B4E24','#EECA98','#EBAA4F','#6B5B45','#B8853E','#AB891A','#6B5610','#EBBC23','#F7B50F']

def color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("hsl(40, 70%%, %d%%)" % np.random.randint(45,55))

img = plt.imread("blossom.jpg")

wordcloud1 = WordCloud(
        font_path=FONT_PATH,
        background_color = "black",
        random_state = 1,       
        mask = img
)
wordcloud1.generate_from_frequencies(c)
wordcloud1.recolor(color_func = color_func)
wordcloud1.to_image()
```

<img src = '/assets/images/projects/result_3.png' width = 500> <br>


