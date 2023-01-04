---
title: "[Lecture Note] MySQL 04. Function"
toc: true
use_math: true
categories:
  - RDBMS
tags:
  - [MySQL]
---

[이것이 MySQL이다 강의](https://www.youtube.com/watch?v=xKYeJxBTt2E&list=PLVsNizTWUw7Hox7NMhenT-bulldCp9HP9)를 참고하여 정리한 글임을 밝힙니다.

# 조건문 IF / IFNULL / NULLIF / CASE WHEN ESLE END

일부만 가져온 것이고 이 외의 예시는  06. stored procedure에서 함께 나타낼 것입니다.

- 예시 1
```
SELECT IF (100>200, '참이다', '거짓이다');
SELECT IFNULL(NULL, '널이군요'), IFNULL(100, '널이군요');
SELECT NULLIF(100, 100), NULLIF(200,100);

-- 이건 한 column에서 값들 처리할 때 사용하면 좋을 듯
SELECT CASE 10
			WHEN 1 THEN '일'
            WHEN 5 THEN '오'
            WHEN 10 THEN '십'
            ELSE '모름'
		END AS 'CASE 연습';

```

- 예시 2

```
SELECT U.userID, U.name, SUM(price*amount) AS '총구매액',
       CASE  
           WHEN (SUM(price*amount)  >= 1500) THEN '최우수고객'
           WHEN (SUM(price*amount)  >= 1000) THEN '우수고객'
           WHEN (SUM(price*amount) >= 1 ) THEN '일반고객'
           ELSE '유령고객'
        END AS '고객등급'
   FROM buytbl B
      RIGHT OUTER JOIN usertbl U
         ON B.userID = U.userID
   GROUP BY U.userID, U.name 
   ORDER BY sum(price*amount) DESC ;
```

# 문자열 처리

CONCAT_WS / ELT FIELD FIND_IN_SET, INSTR, LOCATE / INSERT /	LEFT RIGHT / UPPER LOWER / LPAD RPAD / TRIM / REPEAT / REPLACE / SUBSTRING
```
SELECT concat_ws('/', '2025', '01', '01');
SELECT ELT(2, '하나', '둘', '셋'), FIELD('둘', '하나', '둘', '셋'), 
	FIND_IN_SET('둘', '하나', '둘', '셋'), INSTR('하나둘셋', '둘'), LOCATE('둘', '하나둘셋');

SELECT INSERT('abcdefghi', 3, 4,'@@@@'), INSERT('abcdefghi', 3, 2, '@@@@');
SELECT LEFT('abcdefghi', 3), RIGHT('abcdefghi', 3);

SELECT LPAD('이것이', 5, '#'), RPAD('이것이', 5, '##');
```

# 수학 함수

ABS / CEILING, FLOOR, ROUND / SQRT / RAND / SIGN

# 날짜시간 함수

ADDDATE SUBDATE / ADDTIME SUBTIME / CURDATE CURTIME NOW /	YEAR MONTH HOUR MINUTE / DATEDIFF TIMEDIFF /  DAYOFWEEK MONTHNAME DAYOFYEAR / LAST_DAY / PERIOD_ADD

```
SELECT ADDDATE('2025-01-01', INTERVAL 31 DAY), 
	ADDDATE('2025-01-01', INTERVAL 1 MONTH);
SELECT SUBDATE('2025-01-01', INTERVAL 31 DAY), 
	SUBDATE('2025-01-01', INTERVAL 1 MONTH);
    
SELECT DATEDIFF('2025-01-01', NOW()), TIMEDIFF('23:23:59', '12:11:10');

SELECT PERIOD_ADD(202501, 1), PERIOD_DIFF(202501, 202312);
```