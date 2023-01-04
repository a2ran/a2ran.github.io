---
title: "[Lecture Note] 객체지향 프로그래밍 - 12 ArrayList class"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

객체지향프로그래밍 입문 [강의](https://www.inflearn.com/course/%EC%9E%90%EB%B0%94-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9E%85%EB%AC%B8/dashboard)의 강의 내용을 참고하여 정리한 글입니다.


# ArrayList 클래스

기존의 배열은 fixed length가 정해져 있어서 필요하다면 복사해서 크기를 늘려주어야 했다. <br>
혹은 값을 수정 변경할 경우 Array는 순차적으로 저장되는 구조이기 때문에 다 조정해야했다. <br>
이러한 불편함들을 개선한 것이 ArrayList class다.

디테일한 내용은 help에서 보면 좋고, class들이 어떤 구조를 가지고 있는지 알 수 있어서 추후에 찾아보는 것을 추천한다. 특히 method를 외우지 말고 보고 확인하면서 습득해도 될 것 같다.

# 선언

최근에는 generic E를 기본적으로 적는다고 한다. 또한 element는 선언할 때 추가하는 것이 아니라 method로 관리하게 된다.
```
ArrayList<E> <변수명> = new ArrayList<E>();
```

# Method

|Method|설명|
|:---:|:---:|
|boolean add(E, e)|자료형 E인 요소 e를 배열에 추가|
|int size()|배열에 추가된 요소 전체 개수를 반환|
|E get(int index)|index위치의 요소를 반환|
|E remove(int index)|index위치의 요소를 제거하고 반환(ex. pop)|
|boolean isEmpty()|배열이 비어있는지 확인|
|\vdots|\vdots|

# 주의사항

- ArrayList에서는 index 연산자 제공하지 않고 순수 array에서만 제공하게 된다.

# Generic

해당 부분은 나중에 추가해서 다뤄야 할 것으로 생각한다.<br>
보통 어떤 자료형들로 구성되어 있는지 알려주는 것으로 생각해도 될 것 같다.