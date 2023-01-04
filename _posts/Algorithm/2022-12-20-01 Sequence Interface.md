---
title: "[Lecture Summary] 01 Algorithm and Data Structure : Sequence Interface"
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

# API 

## Operations

Sequence Interface should support following operations

![제목](/assets/images/algorithm/1-0.PNG){: width="70%" height="55%"}{: .align-center}

## Properites

- Surely, performance will be different based on implemenation(or algorithm). 
- In the case of build() or Container, it takes time proportional to tie space or size
  - This is about **memory allocation model**. 
- Static Operations take constant time!


# Data Structure

![제목](/assets/images/algorithm/1-1.PNG){: width="70%" height="55%"}{: .align-center}

## (Static) Array Sequence

=> consecutive chunck of momory <br>
=> array[i] = memory[address(array)+i] <br>
= memory[id(array)+i] @ Python

### Properties

0. Size is constant!! 
1. Great at static operation: $O(1)$
2. Poor at dynamic operations: $O(n)$
  - reallocating the array
  - shifting all items
3. Insert_at , delete_at : $\theta(n)$


## Dynamic Array Sequence

= list @ Python<br>

For **empty array**, it resize every 1, 2, 4, 8, ... $\log(n)$. This means that allocating every item at an empty dynamic array is $O(1+2+...+\log(n))$ which is domiated by **last term**!.Therefore, allocating whole items on empty array  $\theta(2^{\log(n)})=\theta(n)$.

Then, allocating a single item on array is **on average** $\theta(1)$ amortized time! (You can see amortize analysis). That's why insert/delete_last is 1 for the above table.

(This is just the case of list @ Python. append and pop are amortized $O(1)$ time but any other operation takes $O(n)$)

### Properties

1. relax constraint on size
2. enforce size = $\theta(n) \ge n$
3. Allocate new array of $2\times size$
4. size = size of the array
5. len = length of item  

## Linked List Sequence

Concept of Point arise!

Every node : an item and a pointer to the next node

### Properties

1. insert/delete_first: $\theta(1)$
2. get/set_at : $\theta(i)$
  - in the worst case, $\theta(n)$
