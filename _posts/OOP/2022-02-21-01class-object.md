---
title: "[Lecture Note] 객체지향 프로그래밍 - 01 클래스와 객체 1 (1)"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

객체지향프로그래밍 입문 [강의](https://www.inflearn.com/course/%EC%9E%90%EB%B0%94-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9E%85%EB%AC%B8/dashboard)의 강의 내용을 참고하여 정리한 글입니다.



# 객체란

사전적으로 *의사나 행위가 미치는 대상*이라고 하며 구체적, 추상적인 데이터 단위로 생각할 수 있습니다.

그렇다면 **객체지향 프로그래밍**(Object Oriented Programming, OOP)는 객체 기반의 프로그래밍이라고 할 수 있습니다.<br>
참고) 절차지향 프로그래밍(Procedural Programming)이 그 반대가 되며 C언어가 있습니다.

- 예시

학교를 간다. <br>
= 아침에 일어나서 / 밥먹고 / 씻고 / 옷을 갈아입고 / 버스를 타고 / ....

이 흐름은 시간(순차)에 따른 프로그래밍으로 생각할 수 있습니다.

학교를 간다<br>
-> 객체 : 학생 / 밥 / 버스 / 학교 <br>
-> 기능 : 일어난다 / 먹는다 / 탄다 / 이동한다.<br>
-> 협력 : 학생이 밥을 먹는다. / ...


# 클래스(Class)

객체에 대한 **속성**과 **기능**을 코드로 구현한 것으로, 일종의 청사진으로 생각

- 객체의 속성 : 특성(property), 속성(attribute), 멤버 변수(member variable)
- 객체의 기능 : method, member function(C++)

예) 학생 <br>
속성 : 학번, 이름, 학년, 사는 곳 등등
기능 : 수강신청, 수업듣기, 시험 보기

## 클래스 기본 구조

```
(접근 제어자, eg. public) class Class_name{
  member variable;
  method;
  ...
}
```

특징
- 보통 클래스 이름은 대문자로 시작
- 하나의 java 파일에는 하나의 public class와 여러개의 클래스가 같이 있을 수 있습니다.
- 단 public class의 이름과 자바 파일 이름은 동일해야 합니다.


 ## 작동

 일종의 웹서비스라고 생각하면 서버에 요청이 들어가서 처리할 때 서블릿이 엔트리포인트(시작점)이 되는 것이지만<br>
 지금 여기서 시작이 되는 method는 것이 main이며 형식은 동일합니다.<br>
 이 main함수는 method라고 불리지 않고 JVM이 호출하여 시작하는 함수입니다.

 ```
 public static void main(String[] args){
   ...
 }
 ```

# 자바 프로젝트 기본 구조

**패키지**는 소스의 묶음으로 두기 위해 만드는 것이고 이름의 역할은 곧 소스의 성격을 나타내게 됩니다. 이렇게 정리해서 소스의 계층구조를 나타내고 유지 보수에 맞춰서 작성하게 됩니다.

클래스의 이름의 원래 이름은 *package_name.class_name*이 됩니다. 그래서 클래스를 구분할 수 있게 되고 디스크에는 디렉토리로 구분됩니다.



# 예시

- 기본 예시 1 : 한 클래스에 다 작성

```
public class Student {
	int studentID;
	String studentName;
	int grad;
	String address;
	
	public void showStudentInfo() {
		System.out.println(studentName+","+address);
	}
	
	public static void main(String[] args) {
		Student studentLee = new Student();
		studentLee.studentName = "이순신";
		studentLee.address = "서울시 서초구 서초동";
		
		studentLee.showStudentInfo();
	}
}
```

- 기본 예시 2 : main이 있는 다른 class를 만들고 Student class 구분해서 작동

```
public class Student {
	int studentID;
	String studentName;
	int grad;
	String address;
	public void showStudentInfo() {
		System.out.println(studentName+","+address);
	}
}

// 실행은 아래 클래스를 실행
public class StudentTest {
	public static void main(String[] args) {
		Student studentLee = new Student();
		studentLee.studentName = "이순신";
		studentLee.address = "서울시 서초구 서초동";
		
		studentLee.showStudentInfo();
	}
}

```
