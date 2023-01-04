---
title: "[Lecture Note] MySQL 09.Cursor"
toc: true
use_math: true
categories:
  - RDBMS
tags:
  - [MySQL]
---

[이것이 MySQL이다 강의](https://www.youtube.com/watch?v=xKYeJxBTt2E&list=PLVsNizTWUw7Hox7NMhenT-bulldCp9HP9)를 참고하여 정리한 글임을 밝힙니다.


stored procedure 내에서 사용하는 cursor에 대한 내용입니다.

커서의 핵심은 파일을 열어서 한 행식 값을 처리하는 것을 말하고 이 때 '파일 포인터'는 자동으로 다음 줄을 가리켜 진행됩니다.

# 커서의 처리 순서

1. 커서의 선언
 - DECLARE CURSOR
2. 반족 조건 선언
 - DECLARE CONTINUE HANDLER
 - 더이상 읽을 것이 없을 때 실행할 내용 지정
3. 커서 열기
 - OPEN
4. 커서에서 데이터 가져오기
 - FETCH
5. 데이터 처리
 - 4,5번 단계를 LOOP END LOOP 문으로 반복실행
6. 커서 닫기
 - CLOSE

 
- 예시 1
```
DELIMITER $$
CREATE PROCEDURE cursorProc()
BEGIN
    DECLARE userHeight INT; 
    -- 파이썬에서 count하는 것과 비슷한 맥락의 변수
    DECLARE cnt INT DEFAULT 0; 
    DECLARE totalHeight INT DEFAULT 0;
    
    -- 마지막 행인지 파악하는 bool값
    DECLARE endOfRow BOOLEAN DEFAULT FALSE;

    -- 1. FOR 뒤에 오는 테이블에서 커서가 진행됩니다. 
    DECLARE userCuror CURSOR FOR SELECT height FROM userTbl;

    -- 2. CONTINUE이기 때문에 "NOT FOUND" 에러가 발생한 경우 endOfRow에 TRUE를 대입하는 방식의 에러 처리
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET endOfRow = TRUE;
    
    -- 3. 
    OPEN userCuror;

    -- 5.
    cursor_loop: LOOP
        -- 4. 
        -- 커서가 다루는 데이터(userCuror)를 userHeight라는 이전에 선언된 변수에 대입
        FETCH  userCuror INTO userHeight; 
        
        -- break 문과 같은 역할
        IF endOfRow THEN 
            LEAVE cursor_loop;
        END IF;

        SET cnt = cnt + 1;
        SET totalHeight = totalHeight + userHeight;        
    END LOOP cursor_loop;
    
    SELECT CONCAT('고객 키의 평균 ==> ', (totalHeight/cnt));
    
    CLOSE userCuror;  -- 커서 닫기
END $$
DELIMITER ;

CALL cursorProc();
```

- 예시 2
```
DELIMITER $$
CREATE PROCEDURE gradeProc()
BEGIN
    DECLARE id VARCHAR(10);
    DECLARE hap BIGINT; 
    DECLARE userGrade CHAR(5); 
    
    DECLARE endOfRow BOOLEAN DEFAULT FALSE; 

    -- 여러 값을 가져올 수 있습니다.
    DECLARE userCuror CURSOR FOR
        SELECT U.userid, sum(price*amount)
            FROM buyTbl B
                RIGHT OUTER JOIN userTbl U
                ON B.userid = U.userid
            GROUP BY U.userid, U.name ;

    DECLARE CONTINUE HANDLER FOR NOT FOUND SET endOfRow = TRUE;
    
    OPEN userCuror;  
    grade_loop: LOOP
        -- 여러가지 값을 사용하니 한번에 각각 나눠서 대입하게 됩니다.
        FETCH  userCuror INTO id, hap; 
        
        IF endOfRow THEN
            LEAVE grade_loop;
        END IF;

        CASE  
            WHEN (hap >= 1500) THEN SET userGrade = '최우수고객';
            WHEN (hap  >= 1000) THEN SET userGrade ='우수고객';
            WHEN (hap >= 1) THEN SET userGrade ='일반고객';
            ELSE SET userGrade ='유령고객';
         END CASE;
        
        UPDATE userTbl SET grade = userGrade WHERE userID = id;
    END LOOP grade_loop;
    
    CLOSE userCuror;  -- 커서 닫기
END $$
DELIMITER ;
```