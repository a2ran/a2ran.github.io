---
title: "[Lecture Note] 객체지향 프로그래밍 - 19 인터페이스 활용하기(3)"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

객체지향프로그래밍 입문 [강의](https://www.inflearn.com/course/%EC%9E%90%EB%B0%94-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9E%85%EB%AC%B8/dashboard)의 강의 내용을 참고하여 정리한 글입니다.

******

# 인터페이스

- 상수 : 모든 변수는 상수로
- 추상 메서드 : 모든 메서드는 추상 메서드로 구현
- (JAVA 8+)디폴트 메서드 : 기본 구현을 가지는 메서드, 구현 클래스에서 재정의 가능
  -  모든 구현 클래스에서 동일하게 사용하는 상황에 이용가능
- (JAVA 8+)정적 메서드: 인스턴스 생성과 무관히 **인터페이스 타입**으로 사용하는 메서드
  - 인터페이스의 구현이 없어도, 구현 클래스가 없어도 사용할 수 있는 정적 메서드
- (JAVA 8+)private 메서드 : 인터페이스를 구현한 하위 클래스에서 재정의할 수 없는 메서드
  - 인터페이스 내부에서만 기능을 제공하기 위해 구현하는 메서드

```
public interface Calc {
  ...	
	// static은 인스턴스 생성과 무관히 사용할 수 있다.
	static int total(int[] arr) {
		int total = 0;
		
		for (int i: arr) {
			total += i;
		}
		return total;
	}
	
	// private static은 static method에서만, private는 default method에서 사용하면 된다.
}

```

# 인터페이스 2개

자바에서는 다중 상속을 허용하지 않는다. 다만 인터 페이스를 2개 사용할 수 있다.

default method가 중복되면 override를 하고, 이 때 가상 메서드로 처리되어서 다 적용된다.

```
public interface X {
	void x();
}
...
public interface Y {
	void y();
}
...
public interface MyInterface extends X, Y{
	void myMethod();
}
```

# 인터페이스 상속

구현 코드의 상속이 아니므로 타입 상속(type inheritance)이기 때문에 결국 형변환과 연관이 많아 진다.

```
public class MyClass implements MyInterface{

	@Override
	public void x() {
		System.out.println("x()");
	}

	@Override
	public void y() {
		System.out.println("y()");
	}

	@Override
	public void myMethod() {
		System.out.println("myMethod()");
	}
	
	public static void main(String[] args) {
		MyClass myClass = new MyClass();
		
		X xClass = myClass;
		xClass.x();
	}
}

```

# 인터페이스 구현과 클래스 상속 함께 하기