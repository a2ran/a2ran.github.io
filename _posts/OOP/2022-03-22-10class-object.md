---
title: "[Lecture Note] 객체지향 프로그래밍 - 10 객체 배열 사용하기"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

객체지향프로그래밍 입문 [강의](https://www.inflearn.com/course/%EC%9E%90%EB%B0%94-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9E%85%EB%AC%B8/dashboard)의 강의 내용을 참고하여 정리한 글입니다.

# 객체 배열 선언하기

배열을 선언하였을 때 배열에 담기는 것은 객체를 가르키는 주소가 담기는 배열이 되는 것이다.<br>

아래의 그림에서 볼 수 있듯이 주소의 크기는 4이든 8바이트이든 가능하지만, 중요한 것은 나중에 생성해야 하는 것이다.

![제목](/assets/images/oop/arraylist1.png){: width="50%" height="40%"}{: .align-center}

# 배열 복사하기

## 용도

처음에 개수를 정하고 사용하기에 더 큰 배열이 필요한 경우 기존 배열의 값을 복사하게 된다.

## 방법
```
System.arraycopy(src, srcPos, dest, destPos, length);
```

|변수 유형|선언 위치|
|:---:|:---:|
|src|복사할 배열 이름|
|srcPos|복사할 배열의 첫번째 위치|
|dest|복사해서 붙여 넣을 대상 배열 이름|
|destPost|복사해서 대상 배열에 붙여 넣기를 시작할 첫번째 위치|
|length|src에서 dest로 자료를 복사할 요소 개수|

## 객체 배열 복사

### 얕은 복사
단순히 복사할 때엔 객체가 복사된 것이 아니라 **주소가 복사**된 것이다.

![제목](/assets/images/oop/arraylist2.png){: width="50%" height="40%"}{: .align-center}

### 깊은 복사
아래와 같이다시 생성해야 한다.
![제목](/assets/images/oop/arraylist3.png){: width="50%" height="40%"}{: .align-center}

# enhanced for loop

배열의 요소의 처음부터 끝까지 모든 요소를 참조할 때 사용하는 for loop

```
for (<type> 변수명 : 배열){
  반복 실행문;
}
```
