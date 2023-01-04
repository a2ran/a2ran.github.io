---
title: "[Lecture Note] 객체지향 프로그래밍 - 자료형 String"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

원문 [링크](https://wikidocs.net/205)이며 해당 글을 참고한 정리한 글입니다.


# String

기존에 알고 있던 char 타입은 문자를 나타낸 것이라면 문자열(String)은 이런 문자들이 나열된 것으로 생각할 수 있습니다. 

중요한 것은 따옴표는 char를 쌍따옴표는 String의 리터럴이 됩니다.

## Declarartion

- 리터럴 표기

일반적인 변수 선언처럼 나타낼 수 있으며 객체생성없이 String Constant pool에 저장하여 사용하게 됩니다.

```
String a = "Happy Java";
String b = "a";
String c = "123";
```

- 객체로 표기

Heap 영역에 항상 새로운 객체를 만들게 됩니다.

```
String a = new String("Happy Java");
String b = new String("a");
String c = new String("123");
```

객체로 새롭게 초기화해서 사용할 경우 동일한 문자열을 가지고 있어도 서로 다른 객체로 판단될 수 있다는 점이 큰 차이이며

이렇게 차이가 생기는 이유는 데이터를 JVM 작동 원리를 더 자세히 알아보아야 할 것 같습니다. 


## Primitive data type

문자열 이외의 다른 자료형 중에서 리터럴 표기로 변수를 선언하였던 데이터 타입들을 **원시 자료형**이라고 하며 *new*를 이용해 선언할 수 없습니다. 

다만 문자열을 리터럴 표기로 선언하였다고 하더라도 원시 자료형이라고 부르지 않습니다. 

### Wrapper class

각각의 원시 자료형엔 대응하는 wrapper class가 존재하고 다음과 같습니다.

|원시자료형|Wrapper 클래스|
|:---:|:---:|
|int|Integer|
|long|Long|
|double|Double|
|float|Float|
|boolean|Boolean|
|char|Char|

이후에 다룰 자료형 중에서 값이 아닌 객체를 다룰 때 Wrapper class를 사용한다고 합니다. 


## method

- ==, equals()

앞서 데이터들이 저장되는 방식이 [선언 방식](#declarartion)에 따라 달라졌기 때문에 

등호 비교할 때에도 **같은 값을 가져도 다른 객체**라면 false가 리턴될 수 있습니다.


다음 두 객체에 대해 비교한 결과를 정리하면 다음과 같습니다.

```
String co = new String("a");
String ct = "a";
```

|비교방법|결과|
|:---:|:---:|
|co==ct|false|
|co.equals(ct)|true|

이런 결과가 나오는 이유는 equals의 경우 값을 읽어서 비교하는 반면 "=="는 동일한 객체인지 판단하는 기호라고 합니다.

자세한 둘의 비교는 [다음 글](https://hyeran-story.tistory.com/123)을 참고하였습니다.


그 외의 method는 다음과 같습니다.

- indexOf : argwhere처럼 문자열(input)이 시작하는 인덱스(output)를 알려줍니다.
- contains : pd.Series.str.contains처럼 특정 문자열(input)의 포함 여부를 bool형(output)으로 리턴합니다.
- charAt : indexOf와 비슷하지만 파이썬의 String indexing처럼 특정한 하나의 문자의 위치(input)의 문자(output, **char**)를 알려주게 됩니다
- replaceAll : pd.Series.str.replace처럼 주어진 문자 패턴(input1)을 다른 값(input2)으로 바꿔(output)줍니다.
- substring : 파이썬의 String indexing처럼 주어진 인덱스 범위(input)의 문자열(output)을 반환합니다.
- split : pd.Series.str.split과 비슷한 작업입니다.
- toUpperCase / toLowerCase

## Formatting

파이썬에서 f-string과 마찬가지로 문자열 내 정해진 위치에 값을 대입해서 문자열로 리턴하는 방법을 말합니다.

```
-- 문자열을 리턴
System.out.println(String.format("I eat %d apples.", 3)); 
System.out.println(String.format("I eat %s apples.", "five"));
Sysstem.out.println(String.format("I ate %d apples. so I was sick for %s days.", number, day));

-- 문자열을 출력
System.out.printf("I eat %d apples.", 3); 
```

예시는 위와 같으며 과 함께 순서대로 값을 입력해주면 됩니다. 이 때 값을 구분하기 위해 사용하는 패턴이 있습니다.


|코드|설명|
|%s|문자열(String)|
|%c|문자 1개(character)|
|%d|정수(Integer)|
|%f|부동소수(floating-point)|
|%o|8진수|
|%x|16진수|
|%%|Literal % (문자 % 자체)|

주의할 사항은 다음과 같습니다.

- 마지막 %%의 경우 %자체를 사용하기 위해선 \n처럼 앞에 %를 붙여서 사용하게 됩니다.
- %s를 사용하게 된다면 정수 값을 파라미터로 사용하더라도 알아서 String으로 형변환이 되어 사용된다고 합니다.
- %와 코드 사이에 숫자를 넣게 된다면 공백을 의미 ex. %3s => "   "
- 소숫점을 제한할 때엔 %.(원하는길이)f로 표시 ex. %.3f => 3.14729 -> 3.147


# String Buffer & String bulider

참고한 자료는 [링크](https://ifuwanna.tistory.com/221)에서 확인할 수 있습니다.

문자열을 파이썬의 list 조작할 때처럼 사용하는 method를 가지는 클래스입니다.

## append

String 타입의 경우 + 로 추가하는 것을 append method로 구현가능합니다.

```
StringBuffer sb = new StringBuffer();  
sb.append("hello");
sb.append(" ");
sb.append("jump to java");
String result = sb.toString();

String result = "";
result += "hello";
result += " ";
result += "jump to java";
```

StringBuffer의 append의 경우 하나의 객체에서 계속 수정하는 방식이며

String의 +의 경우 계속 객체를 생성하는 방식입니다.


## 그 외 method

- insert : 정해진 위치(input)에 주어진 문자열 추가(output)
- substring : String과 동일


||String|StringBuffer|StringBuilder|
|:---:|:---:|:---:|:---:|
|저장공간|String (Constant) Pool | Heap | Heap|
|수정여부|새로운 객체 생성|하나의 객체 수정|하나의 객체 수정|
|mutability|immutable|mutable|mutable|
|성능|비교적 가벼움|비교적 무겁고 느림|빠름|
|동기화|가능|가능|불가능|
|thread-safe|Y|Y|N|
|예시|변화적고 멀티쓰레드|변화 많고 멀티 쓰레드|변화 많고 단일 쓰레드|
