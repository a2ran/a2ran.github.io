---
title: "[Lecture Summary] 02 Algorithm and Data Structure : Set Interface"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Algorithm
tags:
  - [Algorithm, DataStructure]
---

This contents is based on Lecture of [link](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/syllabus/)

Basically, set interface has following properties.

- Each item has a unique key
- This is a dictionary @ Python.
- $u$ means the size of numbers(or largest key!)

# Operations

Set Interface should support following operations

![제목](/assets/images/algorithm/2-0.PNG){: width="70%" height="55%"}{: .align-center}


# Data Structure

![제목](/assets/images/algorithm/2-1.PNG){: width="70%" height="55%"}{: .align-center}

## (Unordered) Array

Searching which is find(k) is $O(n)$ in the case of $k$ is located at the last index!

For dynamic operations, which is insert/delete function, it is amortized $O(n)$

## Sorted Array

This array is build being organized by key
- "organizing"(or sorting) takes $O(n\log(n))$!
- Once we build it, other operations are much faster.

Searching operation is $O(\log(n))$ by binary search!
- find(k) and find_prev/find_next time complexity is same.



