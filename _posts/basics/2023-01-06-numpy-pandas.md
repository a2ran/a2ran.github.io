---
title: "[basics] numpy & pandas syntax"
toc: true
use_math: true
categories:
  - basics
tags:
  - [basics, syntax]
date: 2023-01-06
last_modified_at: 2023-01-06
sitemap:
  changefreq: daily
  priority: 1.0
---

Essential Numpy and Pandas syntax

# <span style = "color : skyblue"> numpy </span>

basic numpy library syntax on python

## <span style = "color : blue">Array:</span>
- All elements require same data types.
  - \# of elements in the internal array must be equal
    - Array performs operation between elements
    
```python
print(np.array([1, 3, 5, 'a', 'b'])) # all elements are changed into str
['1' '3' '5' 'a' 'b']
```

<br>

```python
print(np.array([[1], [3, 5], [2, 4, 6]])) # cannot perform
```

<br>

```python
a1 = np.array([1,2,3,4,5]).reshape(5,1)
a2 = np.array([6,7,8,9,10])
print(a1 + a2) # broadcasting
[[ 7  8  9 10 11]
 [ 8  9 10 11 12]
 [ 9 10 11 12 13]
 [10 11 12 13 14]
 [11 12 13 14 15]]
```

## <span style =  "color : blue"> Matrix </span>

```python
# numpy.full(shape, fill_value)
np.full((3, 3), 2)

array([[2, 2, 2],
       [2, 2, 2],
       [2, 2, 2]])
```
<br>

```python
# np.linspace(start, end, num = number,endpoint = True)
np.linspace(0, 2, 5)

array([0. , 0.5, 1. , 1.5, 2. ])
```
<br>

```python
# np.logspace(start, end, num = number, endpoint = True, base = 10.0, dtype = object)
np.logspace(0, 4, 4, endpoint = False)

array([   1.,   10.,  100., 1000.])
```

# <span style = "color : skyblue"> pandas </span>

basic pandas library syntax on python through titanic_dataset

![image](https://blog.kakaocdn.net/dn/N4NAc/btqRoP1ml8o/x4DD7ITezrXVcJKgEYR5a1/img.png)

## <span style = "color : blue"> titanic_dataset </span>

```python
# import titanic dataset
titanic_df = pd.read_csv('/content/drive/MyDrive/Titanic.csv', index_col=0)
```
<br>

```python
# sums up num of NaN in each column
titanic_df.isnull().sum()

Pclass        0
Name          0
Sex           0
Age          86
SibSp         0
Parch         0
Ticket        0
Fare          1
Cabin       327
Embarked      0
dtype: int64
```

```python
#df.sort_values(by, *, axis = 0, ascending = True, inplace = False)
titanic_df.sort_values(by = ['Pclass','Age'], ascending=(False, False), inplace = False)
```

## <span style = "color: blue"> boolean indexing </span>

```python
# Extract desired conditions using "loc" to specify rows and columns
titanic_df.loc[titanic_df['Pclass'] == 1, 'Name]
```

|**PassengerId**|Name|Pclass|
|:---:|---|---|
|892|Kelly, Mr. James|3|
|893|Wilkes, Mrs. James (Ellen Needs)|3|

## <span style = "color : blue"> aggfunc </span>

```python
# groupby.agg(): group variables by set functions
df5.groupby(['class']).agg({'score':['sum', 'min', 'max'],
                            'exam':'count'})
```
                            
|**class**|sum|min|max|count|
|:---:|---|---|---|---|
|1|124|10|50|4|
|2|89|10|34|4|
