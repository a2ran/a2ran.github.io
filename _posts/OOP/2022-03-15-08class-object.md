---
title: "[Lecture Note] 객체지향 프로그래밍 - 08 클래스와 객체 2 (4) - singleton 패턴"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

객체지향프로그래밍 입문 [강의](https://www.inflearn.com/course/%EC%9E%90%EB%B0%94-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9E%85%EB%AC%B8/dashboard)의 강의 내용을 참고하여 정리한 글입니다.


프로그램 내부에서 **유일**하게 하나만 생성되어야 하는 경우에 singleton 패턴을 사용하게 된다. <br>
C언어에서는 global로 선언하면 되지만, 자바의 경우 외부에서는 클래스 형태로 선언되는 것만을 허용한다.
- 여러 다른 소스코드에서 각자 서로 다른 instance를 생성해버리면 안되는 것이다.

# singleton 패턴

1. 자바에서 global 변수를 선언할 수 없기 때문에 static 변수를 사용하며,
  - static은 데이터 영역에 저장되어 공유한다는 성질을 이용
2. 생성자를 private으로 만들고,
  - 이렇게 하면 외부에서 constructor에 접근이 안되 함부로 생성할 수 없다.
3. public의 static method를 사용하여 접근가능하도록 한다.
  - static methdo이기 때문에 클래스 생성 무관히 사용할 수 있게 된다.

# 구조

|클래스이름|Singleton|
|:---:|:---:|
|멤버 변수|- instance|
|생성자와 metod|- Singleton + getInstance|