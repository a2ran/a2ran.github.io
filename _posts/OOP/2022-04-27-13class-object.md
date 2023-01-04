---
title: "[Lecture Note] 객체지향 프로그래밍 - 13 상속과 다형성(1)"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

객체지향프로그래밍 입문 [강의](https://www.inflearn.com/course/%EC%9E%90%EB%B0%94-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9E%85%EB%AC%B8/dashboard)의 강의 내용을 참고하여 정리한 글입니다.

******

# Intro

프로그램 구조가 유연하게 코딩되고, 유지 보수가 수월하게 만들도록 하는 다형성


# 상속

클래스를 정의할 때 이미 구현된 클래스를 상속(inheritance) 받아서 속성이나 기능이 확장되는 클래스를 구현함.

코드를 재사용하긴 하지만 재사용의 방법이 상속이라고는 말하지 않는다. 즉 동일한 속성이 있다고 해서 상속하지 않고 논리적으로 포함관계에 있어야 할 것 같다. 이런 관계를 **is-a** 관계라고 부르며 반대로 단순히 코드가 재사용되는건 **has-a** 관계라고 부른다.

참고) 재사용 방법은 상속과 합성(aggregation)이 있다.

## 상속하는 클래스

: 상위 클래스, parent class, base class, super class

주로 하위 클래스보다 일반적인 의미를 가짐

## 상속받는 클래스

: 하위 클래스, child class, derived class, subclass

주로 상위 클래스보다 구체적인 의미를 가짐

# 특징

- 상위 클래스에서 private으로 선언되어 있으면 visiability가 없다. 즉 하위 클래스에서 접근할 수 없게 된다. 

- protected로 사용하게 된다면 상속관계에서는 public으로 사용할 수 있게 된다. 이런 경우 package가 달라도 public으로 사용할 수 있다. 

- 접근 제어자를 사용하지 않으면 default 상태이기 때문에 같은 package에서만 보이게 된다. 이런 경우 set이나 get function으로 제어가능할 수 있다.

- 가시성 순서 : private - default - protected - public

# 문법 

```
class B extends A{

}
```

# 예시

```
class Mammal{}

class Human extends Mammal{}
```
