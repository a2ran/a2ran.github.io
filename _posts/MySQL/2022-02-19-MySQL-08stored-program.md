---
title: "[Lecture Note] MySQL 08. Stored Program"
toc: true
use_math: true
categories:
  - RDBMS
tags:
  - [MySQL]
---

[이것이 MySQL이다 강의](https://www.youtube.com/watch?v=xKYeJxBTt2E&list=PLVsNizTWUw7Hox7NMhenT-bulldCp9HP9)를 참고하여 정리한 글임을 밝힙니다.


# 기본 형태 

내장함수만으론 부족한 것을 채워주고 사용법은 stroed procedure와 조금 다릅니다.

특히 함수이기 때문에 반환값이 필수적으로 있어야 합니다.


호출할 때엔 SELECT를 사용하여 반환 값을 사용할 수 있습니다.

입력파라미터만 있고 출력문으로 형식과 값을 지정할 수 있게 됩니다. 

또한 src안에 SELECT를 사용할 수 없습니다.

```
DELIMITER $$
CREATE FUNCTION stored_program_name(parameter_name parater_type, ...)
  RETURNS return_type
BEGIN
  (src)
  RETURN return_value
END $$
DELIMITER;

SELECT stored_program_name();
```

또한 Stored_program 사용을 위해선 시작 전에 생성 권한을 주어야 합니다.

```
SET GLOBAL log_bin_trust_function_creators = 1;
```


- 예시 1

```
DELIMITER $$
CREATE FUNCTION getAgeFunc(bYear INT)
    RETURNS INT
BEGIN
    DECLARE age INT;
    SET age = YEAR(CURDATE()) - bYear;
    RETURN age;
END $$
DELIMITER ;

SELECT getAgeFunc(1979);
```

- 예시 2 결과값을 변수로 사용
SELECT INTO문으로 결과 값을 변수로 사용 가능합니다.
또한 하나의 expr로 사용할 수 있습니다.

```
SELECT getAgeFunc(1979) INTO @age1979;
SELECT getAgeFunc(1997) INTO @age1989;
SELECT CONCAT('1997년과 1979년의 나이차 ==> ', (@age1979-@age1989));
```

# 기타

함수 내용 확인 코드

```
SHOW CREATE FUNCTION getAgeFunc;
```

