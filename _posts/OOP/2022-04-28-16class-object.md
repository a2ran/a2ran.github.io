---
title: "[Lecture Note] 객체지향 프로그래밍 - 16 다형성 활용가 다운캐스팅(4)"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

객체지향프로그래밍 입문 [강의](https://www.inflearn.com/course/%EC%9E%90%EB%B0%94-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9E%85%EB%AC%B8/dashboard)의 강의 내용을 참고하여 정리한 글입니다.

******

항상 상속을 사용해야하는 것은 아니며 아래의 관계를 고려하여 코딩하자.

![제목](/assets/images/oop/16-polymorphism.jpg){: width="50%" height="40%"}{: .align-center}


# 다운 캐스팅(instanceof)

명시적 형변환을 통해서 다운 캐스팅이 되어야 하고, 타입을 확인하는 예약어로 instanceof를 사용하게 된다.

주로 다운 object 클래스로 리턴할 때 다운캐스팅하여 사용하기도 한다.
