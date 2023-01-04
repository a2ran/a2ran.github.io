---
title: "[Lecture Note] 객체지향 프로그래밍 - 03 클래스와 객체 1 (3)"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

객체지향프로그래밍 입문 [강의](https://www.inflearn.com/course/%EC%9E%90%EB%B0%94-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9E%85%EB%AC%B8/dashboard)의 강의 내용을 참고하여 정리한 글입니다.

# 용어

|용어|설명|
|:---:|:---:|
|객체| 생성된 인스턴스|
|클래스|코드로 만든 상태|
|인스턴스|클래스가 메모리에 생성된 상태|
|멤버 변수|클래스의 속성|
|메서드|클래스의 기능|
|참조 변수|메모리에 생성된 인스턴스를 가리키는 변수|
|참조값|인스턴스 메모리 주소값|

# Class와 Instance

흐름은 다음과 같습니다.

1. 클래스(static 코드) 선언
2. 생성(인스턴스 화)
3. 인스턴스(dynamic memory)

# 생성
예약어 new를 이용해서 생성하게 됩니다.<br>
함수처럼 보이는 생성자를 이용해서 생성하게 됩니다. <br>

웹서버든 안드로이드든 각자의 방식, 순서대로 선언하게 되고 우리는 간단한 형태로 다뤄보기로 합니다.

- 기본 틀
```
class_type var_name = new constructor
Student studentA = new Student();
```

형태가 일반적인 int, char 등의 변수들을 지정하는 것처럼 보이는데 사실 기존의 데이터 타입들은 **primitive data type**으로 원래 언어에서 제공하는 것들입니다.

하지만 이건 **참조형 데이터 타입**이고 속성을 정할 수 있고, 객체는 생성해서 사용하게 됩니다.

그리고 이 instance가 할당된 변수는 **reference variable**이라고 부릅니다. 이 변수를 통해서 참조할 수 있는 값들을 확인할 수 있습니다.

# Instance & Heap

지역변수들이 자리를 잡는 곳이 stack 메모리이고, 객체를 생성하게 되면 instance는 Heap메모리 영역에 위치하게 됩니다. 이 때 속성값들이 여기에 존재한다는 것이고 method는 다른 곳에 있다고 합니다.

즉 지역변수들(혹은 참조 변수)은 stack 메모리에 있으며 각 지역 변수들은 heap메모리에 있는 instance를 가리키고 있습니다. 

heap메모리는 **동적**메모리라고 불리며 필요할 때 allocation을 받는다고 합니다. 없애는 과정은 C++에선 직접 없애야 했지만 자바에선 garbage collector(GC)가 없앤다고 합니다. 

이와 달리 stack 메모리는 함수가 실행되고 있을 때 올라가고 함수가 끝나면 사라지게 됩니다.

## 참조값

실제 참조변수를 print한 경우 나오는 메모리 주소값을 말합니다.
```
package_name.reference_type@16진수(32bit)주소
```

# 생성자
new 예약어로 호출되는 것으로 인스턴스를 생성하면서 해야할 일들을 구현한 것을 말합니다.

```
<modifiers; public, private> <class_name>([<argument_list>]){
	[<statements>]
}
```

- default 생성자

자바 컴파일러가 생성자가 없는 경우 pre-compile단계에서 추가해줍니다.<br>
매개변수와 구현부 모두 없다는 것이 특징입니다.
```
<modifiers; public, private> <class_name>(){}
```

- 생성자 오버로드
직접 생성자를 만들어 줄 수 있게 되고 default 생성자도 사용하고 싶다면 직접 입력하면 됩니다. 이 때 생성자들의 이름은 같아도 되지만 매개변수가 달라야하며 이것을 생성자 오버로딩(**constructor overloading**)이라고 합니다.

특징
- 이름은 클래스 이름과 동일해야 한다.
- method가 아니고, 상속도 되지 않으며 리턴값은 없습니다. 

# Implementation

흔히 main함수는 class의 method가 되는 것이 아니라 JVM이 호출하게 되는 함수로 구분해야 합니다.

그래서 앞으로 실행하는 class와 객체를 담는 클래스는 구분해서 사용하기로 합니다.
