---
title: "[basics] Markdown basic syntax"
toc: true
use_math: true
categories:
  - basics
tags:
  - [basics, syntax]
date: 2023-01-04
last_modified_at: 2023-01-04
sitemap:
  changefreq: daily
  priority: 1.0
---

Essential markdown syntax for github blogs.

# Pharaphrasing

## Notice Feature
By using \<div class> html tag, it is possible to call Notice class

```
<div class="notice--primary" markdown="1">
code section is also available
    ```python
print("Hello World!")
    ``` 

- Hello World!
</div>
```

<div class="notice--primary" markdown="1">
code section is also available
    ```python
print("Hello World!")
    ``` 

- Hello World!
</div>


## break
add **\<br>** whenever new lines are needed.

```
Hello <br> World!
```
Hello <br> World!

## Overlapping Structure
Press space bar twice for second line, four times for third line.

```
- Hello
  - World
    - !!!
```

- Hello
  - World
    - !!!

# Text

## *Italics*

```
*Italics*
```
*Italics*

<br>

```
***Italics and Bold***
```

***Italics and Bold***

## <u>Underline</u>

```
<u>Underline</u>
```
<u>Underline</u>

## <span style = "color: blue">colors</span>

```
<span style = "color: red">Red</span>
```

<span style = "color: red">Red</span>

# links
## inline links
inline links with explanation

```
[Google](https://www.google.com)
```

[Google](https://www.google.com)

## image links
send to link location upon clicking the image

```
![image][('/assets/image/ds_lab.png')](https://www.google.com)
```

![image][('/assets/image/ds_lab.png')](https://www.google.com)


# Others

## Reference
can be expressed by \>. \>> when overlapping

```
> Knowledge is Power
  >> Francis Bacon
```

> Knowledge is Power
  >> Francis Bacon

## Checklists

```
- [ ] Unchecked
- [X] Check
```

- [ ] Unchecked
- [X] Check


## Table

```
|**Title**|Ratings|Opinions|<br>
|:---:|---:|---|<br>
|Avatar|⭐⭐⭐⭐⭐|Good|<br>
|Avengers|⭐⭐⭐⭐⭐|Better|<br>
|Asylum|⭐⭐⭐⭐|Best!!|<br><br>
```

|**Title**|Ratings|Opinions|
|:---:|---:|---|
|Avatar|⭐⭐⭐⭐⭐|Good|
|Avengers|⭐⭐⭐⭐⭐|Better|
|Asylum|⭐⭐⭐⭐|Best!!|

## Toggle List

```
<details><br>
<summary>Click here</summary><br>
<div markdown="1"><br><br>

😎Hidden Message😎<br><br>

</div><br>
</details><br><br>
```

<details>
<summary><u>Click here</u></summary>
<div markdown="1">       

😎Hidden Message😎

</div>
</details>
