---
title: "[Lecture Note] 객체지향 프로그래밍 - 02 클래스와 객체 1 (2)"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

객체지향프로그래밍 입문 [강의](https://www.inflearn.com/course/%EC%9E%90%EB%B0%94-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%9E%85%EB%AC%B8/dashboard)의 강의 내용을 참고하여 정리한 글입니다.



# Method란

함수의 일종으로 하나의 (객체의) 기능을 실행하기 위해 객체 *내부*에 구현되는 함수입니다.<br>
추후에 사용될 때 호출하여 사용하게 됩니다.


# 기본구조

```
int[함수 반환형] add[함수이름] (int num1, int num2[매개변수]){
	int result;
	result = num1 + num2;
	return[예약어] result;
}
```

- 함수 이름 : 기능과 관련하여 명명
- 매개 변수 : 함수의 수행을 위해 필요한 변수
- return : 함수 수행 결과를 반환하기 위한 예약어
- 함수 반환형: 반환 값의 자료형을 나타내며 반환값이 없으면 void라고 작성하면 됩니다.


# 메모리 구조 - 스택 메모리

*main*이 호출되고 그 안에 *addNum*이 호출되는 구조입니다.

```
public static void main(String[] args) {
	int num1 = 10;
	int num2 = 30;
	
	int sum = addNum(num1, num2);
	
	System.out.println(sum);
}

public static int addNum(int n1, int n2) {
	int result = n1 + n2;
	return result;
}
```

함수가 사용하는 메모리를 **스택 메모리**라고 합니다.

![사진](/assets/images/oop/02-stack-memory.jpg){: width="50%" height="40%"}{: .align-center}

1. 그래서 처음에 main 함수가 사용하는 변수들(num1, num2, sum)이 먼저 잡히게 됩니다.
2. 그 위에 addNum함수가 사용하는 변수들(n1, n2, result)이 쌓이게 됩니다.
3. 그래서 num1, num2가 n1, n2로 복사가 되고 함수 실행이 끝나면 result가 sum으로 넘어가고 함수가 차지하는 메모리는 사라지게 됩니다.
4. 그렇기 때문에 이름이 같아도 존재하는 공간이 달라서 문제가 없었던 것이고 이렇게 자신의 함수 내에서만 사용되는 변수들을 **지역 변수**라고 부르게 됩니다.


# method 작성

이름을 작성할 땐 작성하는 객체 기준(클라이언트 코드)에서 작성하면 됩니다. 

가령 이름을 알고 싶을 때는 이름을 '가져가게' 되는 것입니다. 변수를 바꿀 때엔 '설정하게' 됩니다.

```
public String getStudentName() {
	return studentName;
}

public void setStudentName(String name) {
	studentName = name;
}
```

# 주의사항

- 효율성 : 필요한 기능을 한번에 유지관리 할 수 있습니다.
- 한 함수에는 한 기능을 이름에 맞게 구현하면 됩니다.