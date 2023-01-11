---
title: "[basics] EDA methodologies"
toc: true
use_math: true
categories:
  - basics
tags:
  - [basics, EDA]
date: 2023-01-11
last_modified_at: 2023-01-11
sitemap:
  changefreq: daily
  priority: 1.0
---

EDA procedures, types, and exercises.

# <span style = "color : blue"> What is Exploratory Data Analysis (EDA)? </span>

Exploratory Data Analysis is the process of analyzing and summarizing a dataset in order to understand its overall structure,
patterns, and relationships. EDA pioneers any data analysis project, and is often used to help formulate hypotheses and identify areas of interest
for further investigation.

## <span style = "color : skyblue"> Data Analysis Procedures </span>

1. Define Problems
- Understand target, define target objectively.
2. Collect Data
- Organize necessary data, identify and secure data location.
3. Data Analysis
- Check for errors, improve data structure and features
4. Data Modeling
- Design data from various views, establish relationships between relative tables
5. Visualization and Re-exploration
- Derive insights to address various types of problem

## <span style = "color : skyblue"> Exploratory Data Analysis Procedures </span>

1. Collect Data
-  Create data collection pipeline, organize required data.
2. Data Preprocessing
- Handle missing data, explore outliers, data labeling...
3. Data Scaling
- Normalize/Standardize data, adjust volume (oversampling/undersampling)
4. Data Visualization
- Data Visualization (Modeling)
5. Post processing
- Explore outliers, Fine Tuning

# <span style = "color : blue"> Exercise </span>

Examples of basic EDA methods

```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```

## <span style = "color : skyblue"> Interpolation </span>

Interpolate missing values using various indicators from the dataframe.

```python
# Interpolate missing values via median
df_train = df_train.fillna(df_train.median())
df_test = df_test.fillna(df_test.median())
```

## <span style = "color : skyblue"> Encoder </span>

Encode string values into int / float scalar to faciliate input selection.

```python
# Interpolate missing values via median
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

enc.fit(train['Sex'])
labels_1 = enc.transform(train['Sex'])
labels_2 = enc.transform(test['Sex'])

train['l_sex'] = labels_1
test['l_sex'] = labels_2
```

## <span style = "color : skyblue"> Histogram, QQplot </span>

Visualize univariate data into histograms and QQplots.

```python
import scipy.stats as stats

for col in numeric_f:
    sns.distplot(filled_train.loc[filled_train[col].notnull(), col])
    plt.title(col)
    plt.show()
```

<img src = '/assets/images/basics/basic_0110_1.png' width = '400'>

```python
from scipy.stats import probplot #for qq plot

f, axes = plt.subplots(2, 4, figsize=(12, 6))
Age = np.array(filled_train['Age'])
Sib = np.array(filled_train['SibSp'])
Par = np.array(filled_train['Parch'])
Age = np.array(filled_train['Fare'])

axes[0][0].boxplot(Age)
probplot(Age, plot=axes[1][0]) #scipy.stats.probplot
axes[0][1].boxplot(Sib)
probplot(Sib, plot=axes[1][1]) #scipy.stats.probplot
axes[0][2].boxplot(Par)
probplot(Par, plot=axes[1][2]) 
axes[0][3].boxplot(Age)
probplot(Age, plot=axes[1][3]) #scipy.stats.probplot

plt.show()    
```

<img src = '/assets/images/basics/basic_0110_2.png' width = '600'>

## <span style = "color : skyblue"> Cross Tabulation </span>

Analyze multivariate data using pandas crosstab library.

```python
pd.crosstab(filled_train['Sex'], filled_train['Pclass'],
            normalize = 'index', margins = True) 
```

## <span style = "color : skyblue"> Scatterplots & Heatmap </span>

Visualize multivariate data using scatterplots and heatmap.

```python
import seaborn as sns

sns.heatmap(df_corr, annot=True)
plt.show()
```

<img src = '/assets/images/basics/basic_0110_3.png' width = '400'>

```python
sns.pairplot(filled_train[list(numeric_f)], 
             x_vars=numeric_f, y_vars=numeric_f)
plt.show()
```

<img src = '/assets/images/basics/basic_0110_4.png' width = '600'>
