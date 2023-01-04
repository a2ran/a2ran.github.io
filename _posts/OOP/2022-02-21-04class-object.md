---
title: "[Lecture Note] 객체지향 프로그래밍 - 04 클래스와 객체 1 (4)"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

객체지향프로그래밍 입문 [강의](https://www.inflearn.com/course/%EC%9E%90%EB%B0%94-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9E%85%EB%AC%B8/dashboard)의 강의 내용을 참고하여 정리한 글입니다.

# Reference data type

변수의 자료형은 크게 기본 자료형(primitive data type)과 참조자료형으로 되어 있습니다.<br>
특히 참조 자료형은 String, Date말고 사용자가 직접 선언한 class도 포함합니다.

한 클래스 내에서 다른 클래스를 가져와서 사용할 때 클래스를 일종의 자료형으로 사용할 수 있게 됩니다.

특히 JDK에서 제공하는 클래스들(ex. String)은 new로 굳이 선언해서 사용할 필요가 없기도 합니다. 

# Information Hiding

객체의 속성을 숨기기 위해 access modifier라고 불리는 **private**이 있습니다.

멤버 변수나 메서드를 외부에서 사용하지 못하도록 해 오류를 예방하며, 변수의 경우엔 get(), set()을 이용해서 접근하도록 합니다.

getter나 setter를 통해서 값을 불러오거나 입력할 때 분기를 시켜서 값을 처리할 수 있게 됩니다.

- default : 같은 패키지 내에서 접근 가능
- public : 외부 접근 허용
- private : 내부에서만 허용
- protected : 상속관계에서만 public처럼 사용가능하고 다른 클래스에선 접근 불가능 