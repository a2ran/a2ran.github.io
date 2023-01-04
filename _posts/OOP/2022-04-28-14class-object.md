---
title: "[Lecture Note] 객체지향 프로그래밍 - 14 상속과 다형성(2)"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

객체지향프로그래밍 입문 [강의](https://www.inflearn.com/course/%EC%9E%90%EB%B0%94-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9E%85%EB%AC%B8/dashboard)의 강의 내용을 참고하여 정리한 글입니다.

******

# 접근 제어자 분류

||외부 클래스|하위 클래스|동일 패키지|내부 클래스|
|:---:|:---:|:---:|:---:|:---:|
|public|O|O|O|O|
|protected|X|O|O|O|
|default|X|X|O|O|
|private|X|X|X|O|

# 클래스 생성 과정

하위 클래스가 생성될 때 
1. 상위 클래스가 먼저 생성됨
2. 싱위 클래스의 생성자가 호출되고 하위 클래스의 생성자가 호출됨
  - 이 때 하위 클래스의 생성자에서는 무조건 상위 클래스의 생성자가 호출되어야 한다.
3. 생성자를 선언하지 않은 경우, 컴파일러가 상위 클래스 기본 생성자를 호출하는 super()를 알아서 넣어준다.
  - super는 this가 자기 자신의 생성자를 호출하는 것처럼 super는 상위클래스의 주소를 가지게 된다.
  - 단, 상위 클래스의 기본 생성자가 없는 경우(혹은 매개변수 생성자만 있는 경우), 하위 클래스는 super()로는 상위 클래스를 생성하지 못하니 명시적으로 상위 클래스를 호출해야 한다.

# 메모리 상태

|힙메모리|어디서|
|:---:|:---:|
|customerID|Customer 클래스의 멤버변수|
|customerName|Customer 클래스의 멤버변수|
|customerGrade|Customer 클래스의 멤버변수|
|bonusPoint|Customer 클래스의 멤버변수|
|bonusRatio|Customer 클래스의 멤버변수|
|agentID|VIPCustomer 클래스의 멤버변수|
|salesRatio|VIPCustomer 클래스의 멤버변수|

중요한 것은 private 멤버변수가 선언되지 않은 것이 아니라 visiability가 없는 것

# 묵시적 형변환(업캐스팅)

상위 클래스의 type으로 변수를 선언하고 이 때 값은 하위 클래스의 instance를 사용할 수 있다. 이것이 가능한 이유는 하위 클래스는 상위 클래스의 type을 내포하고 있기 때문이다. 



# 참고

- 모든 클래스는 Object class를 알아서 상속하기 때문에 기본적으로 super()가 호출되고 있다!