---
title: "[Lecture Note] 객체지향 프로그래밍 - 자료형 Array & list"
toc: true
use_math: true
categories:
  - OOP
tags:
  - [JAVA]
---

원문 [링크](https://wikidocs.net/206)이며 해당 글을 참고한 정리한 글입니다.


# Array

배열(array)이라고 새로운 타입이 있는 것이 아니라 데이터들을 모아둔 것으로 생각하면 됩니다.

## 선언 방식

파이썬에서의 array 혹은 list에선 대괄호를 사용했다라면 여기선 중괄호를 사용해야 합니다.

```
int[] odds = {1, 3, 5, 7, 9};

String[] weeks = new String[7];
weeks[0] = "월";
weeks[1] = "화";
weeks[2] = "수";
weeks[3] = "목";
weeks[4] = "금";
weeks[5] = "토";
weeks[6] = "일";
```

위와 같이 처음에 값들을 다 입력하거나 길이를 지정해놓고 값을 수정하는 방식으로 사용하게 됩니다. 

즉 **길이**라는 요소가 중요하게 작동하며 이를 알려주는 property는 length가 있습니다.


# ArrayList

기존의 배열의 경우 길이라는 요소가 처음에 초기화할 때 필요하였고 바뀌지 않았었는데 이를 동적으로 바꾼 것이 ArrayList라는 자료형입니다.

디테일한 내용(List 인터페이스)은 추후에 다루기로 하고 간단히 생각하자면 List라고 하는 것을 Array의 틀에서 본다고 생각하면 될 것 같습니다.

## Declaration

처음에 import하여 사용해야 합니다.

```
import java.util.ArrayList;
import java.util.Arrays;

ArrayList<String> pitches = new ArrayList<>();  
        pitches.add("138");
        pitches.add("129");
        pitches.add("142");

String[] data = {"138", "129", "142"}; 
        ArrayList<String> pitches = new ArrayList<>(Arrays.asList(data));

ArrayList<String> pitches = new ArrayList<>(Arrays.asList("138", "129", "142"));
```

### Generics
중요한 것은 arraylist에 어떤 값이 들어오는지 초기화할 때 명시해야 합니다. 이를 제네릭스(Generics)라고 합니다.

이렇게 하는 이유로 타입을 알려주지 않으면 Object타입으로 인식하여 arraylist에서 값을 불러올 때 type casting을 해야하게 됩니다.

```
String one = (String) pitches.get(0)
```

ArrayList<String> pitches = new ArrayList<>();

## method

- add : 값(input)을 추가. 출력은 없는 것 같습니다.(void)
- get : 위치(input)의 값(output) 알려줍니다.
- size : length(output) 알려줍니다.
- contains : 특정 값(input)의 포함여부(output)
- remove : 특정 값인 객체나 위치(input)에 해당하는 값 삭제(void)

## application

이렇게 생성산 arraylist를 활용하는 방법은 다음과 같습니다.

- String.join

```
import java.util.ArrayList;
import java.util.Arrays;
...
ArrayList<String> pitches = new ArrayList<>(Arrays.asList("138", "129", "142"));
String[] pitches = new String[]{"138", "129", "142"};

String result = String.join(",", pitches);
```

join을 사용하지 않았더라면 for 문으로 값을 하나씩 불러와 +를 하거나 StringBuffer를 사용해야 했을 것입니다.

- 정렬

Comparator의 naturalOrder()과 reverseOrder()를 이용하여 오름차순/내림차순 정렬을 할 수 있습니다.

```
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

ArrayList<String> pitches = new ArrayList<>(Arrays.asList("138", "129", "142"));
pitches.sort(Comparator.naturalOrder()); 
```




