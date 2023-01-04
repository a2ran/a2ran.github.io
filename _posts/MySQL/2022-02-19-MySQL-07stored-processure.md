---
title: "[Lecture Note] MySQL 07. Stored Procedure"
toc: true
use_math: true
categories:
  - RDBMS
tags:
  - [MySQL]
---

[이것이 MySQL이다 강의](https://www.youtube.com/watch?v=xKYeJxBTt2E&list=PLVsNizTWUw7Hox7NMhenT-bulldCp9HP9)를 참고하여 정리한 글임을 밝힙니다.


SQL에서 프로그래밍하여 사용하는 것으로 일종의 함수, 쿼리문의 집합으로 생각하는 방법인 Stored Procedure에 대한 설명입니다.

# 생성방식

다음과 같은 방식으로 생성합니다.  특히 $말고 %를 사용해도 되는데 중간에 소스코드에서 나오는 ;를 구분하기 위해서 사용하는 것입니다.

특히 프로시저를 만들 때엔 테이블 존재유무를 체크하지 않습니다.

```
DELIMITER $$
CREATE PROCEDURE procedure_name(IN 혹은 OUT 파라미터)
BEGIN
  (소스코드)
END $$
DELIMITER;
```

- 입력 매개변수 사용

value가 input_parameter_name에 할당 됩니다.
```
# 선언시
IN input_parameter_name dtype
...

# 사용시
CALL procedure_name(value)
```

- 출력 매개변수 사용

출력 매개변수에 값을 대입할 때엔 SELECT INTO 문을 주로 사용합니다.

그리고 보통 출력 값을 사용하기 때문에 변수를 출력 매개변수로 사용합니다.

```
# 선언시 
OUT output_parameter_name dtype
...

#사용시
CALL procedure_name(@var_name)
SELECT @var_name;
```

- 예시 1

```
DELIMITER //
CREATE PROCEDURE myProc()
BEGIN	SELECT * FROM memberTBL WHERE memberName = '당탕이';	
SELECT * FROM productTBL WHERE productName = '냉장고';
END //
DELIMITER ; 

CALL myProc() ;
```

이렇게 할 경우 stored procedure의 이름은 *myProc*이 되며 2개의 SELECT문이 실행됩니다. 

선언할 때 주의할 점은 다음과 같습니다.
- 쿼리가 여러개라면 BEGIN-END를 사용합니다
- DELIMITER로 전체 stored procedure문을 다른 쿼리문과 감싸 구분하게 됩니다.

- 예시 2

처음에 선언할 때 userName이란 변수를 선언하는 형태로 생각하면 편할 것 같습니다.

```
DELIMITER $$
CREATE PROCEDURE userProc1(IN userName VARCHAR(10))
BEGIN
  SELECT * FROM userTbl WHERE name = userName; 
END $$
DELIMITER ;

CALL userProc1('조관우');
```

- 예시 3

```
DELIMITER $$
CREATE PROCEDURE userProc3(
    IN txtValue CHAR(10),
    OUT outValue INT
)
BEGIN
  INSERT INTO testTBL VALUES(NULL,txtValue);
  -- id는 testTBL에 있는 열이 되고 가장 큰 값이 outValue가 된다는 것입니다.
  -- 즉 (SELECT FROM) 의 결과가 INTO에 들어가는 형식입니다.
  SELECT MAX(id) INTO outValue FROM testTBL; 
END $$
DELIMITER ;

# 실행
CALL userProc3 ('테스트값', @myValue);
# 실제 사용
SELECT CONCAT('현재 입력된 ID 값 ==>', @myValue);
```


# IF ELSE를 이용

조건문의 형식은 다음과 같습니다.
```
IF condition THEN
  query
ELSE
  query
END IF;
```

- 예시 1

```
DELIMITER $$
CREATE PROCEDURE ifProc()
BEGIN

  DECLARE hireDATE DATE;
  DECLARE curDATE DATE;
  DECLARE days INT;
  
  SELECT hire_date INTO hireDATE 
    FROM employees.employees
    WHERE emp_no = 10001;
    
  SET curDATE = CURRENT_DATE();
  SET days = DATEDIFF(curDATE, hireDATE);

  IF (days/365) >=5  THEN
    SELECT CONCAT('입사한지', days, '일이나 지났습니다.');
  ELSE
    SELECT CONCAT('입사한지', days, '일밖에 안지났습니다.');
  END IF;  
END $$
DELIMITER;
```

- 예시 2

```
DELIMITER $$
CREATE PROCEDURE ifelseProc(
    IN userName VARCHAR(10)
)
BEGIN
    DECLARE bYear INT; -- 변수 선언
    SELECT birthYear into bYear FROM userTbl
        WHERE name = userName;
    IF (bYear >= 1980) THEN
            SELECT '아직 젊군요..';
    ELSE
            SELECT '나이가 지긋하시네요.';
    END IF;
END $$
DELIMITER ;
```

# CASE ~ WHEN 으로 다중 분기

- 예시 1

```
DELIMITER $$
CREATE PROCEDURE caseProc()
BEGIN

  DECLARE point INT;
  DECLARE credit CHAR(1);
  
  SET point = 77;
  
  CASE
    WHEN point >= 90 THEN
      SET CREDIT = 'A';
    WHEN point >= 80 THEN
      SET CREDIT = 'B';
    WHEN point >= 70 THEN
      SET CREDIT = 'C';
    WHEN point >= 60 THEN
      SET CREDIT = 'D';
    ELSE
      SET credit = 'F';
  END CASE;
  SELECT CONCAT('취득점수:', point), CONCAT('학점:', credit);
END $$
DELIMITER;
```

- 예시 2

```
DELIMITER $$
CREATE PROCEDURE caseProc(
    IN userName VARCHAR(10)
)
BEGIN
    DECLARE bYear INT; 
    DECLARE tti  CHAR(3);-- 띠
    SELECT birthYear INTO bYear FROM userTbl
        WHERE name = userName;
    CASE 
        WHEN ( bYear%12 = 0) THEN    SET tti = '원숭이';
        WHEN ( bYear%12 = 1) THEN    SET tti = '닭';
        WHEN ( bYear%12 = 2) THEN    SET tti = '개';
        WHEN ( bYear%12 = 3) THEN    SET tti = '돼지';
        WHEN ( bYear%12 = 4) THEN    SET tti = '쥐';
        WHEN ( bYear%12 = 5) THEN    SET tti = '소';
        WHEN ( bYear%12 = 6) THEN    SET tti = '호랑이';
        WHEN ( bYear%12 = 7) THEN    SET tti = '토끼';
        WHEN ( bYear%12 = 8) THEN    SET tti = '용';
        WHEN ( bYear%12 = 9) THEN    SET tti = '뱀';
        WHEN ( bYear%12 = 10) THEN    SET tti = '말';
        ELSE SET tti = '양';
    END CASE;
    SELECT CONCAT(userName, '의 띠 ==>', tti);
END $$
DELIMITER ;
```

# 반복문 WHILE

- 예시 1

```
DELIMITER $$
CREATE PROCEDURE whileProc()
BEGIN
	DECLARE i INT; 
	DECLARE hap INT; 
    SET i = 1;
    SET hap = 0;

	WHILE (i <= 100) DO
		SET hap = hap + i;  
		SET i = i + 1;      
	END WHILE;

	SELECT hap;   
END $$
```

- 예시 2

```
CREATE TABLE guguTBL (txt VARCHAR(100));

DROP PROCEDURE IF EXISTS whileProc;
DELIMITER $$
CREATE PROCEDURE whileProc()
BEGIN
    DECLARE str VARCHAR(100);
    DECLARE i INT; 
    DECLARE k INT; 
    SET i = 2; 
    
    WHILE (i < 10) DO  
        SET str = ''; 
        SET k = 1; 
        WHILE (k < 10) DO
            SET str = CONCAT(str, '  ', i, 'x', k, '=', i*k);
            SET k = k + 1; 
        END WHILE;
        SET i = i + 1; 
        INSERT INTO guguTBL VALUES(str);
    END WHILE;
END $$
DELIMITER ;
```

# ITERATIVE를 이용한 procedure

ITERATIVE란 예약어를 일종의 continue로,

LEAVE란 예악어는 일종의 break으로 생각하면 됩니다. 

```
DELIMITER $$
CREATE PROCEDURE whileProc2()
BEGIN
    DECLARE i INT; -- 1에서 100까지 증가할 변수
    DECLARE hap INT; -- 더한 값을 누적할 변수
    SET i = 1;
    SET hap = 0;

    myWhile: WHILE (i <= 100) DO  -- While문에 label을 지정
	IF (i%7 = 0) THEN
		SET i = i + 1;     
		ITERATE myWhile; -- 지정한 label문으로 가서 계속 진행
	END IF;
        
        SET hap = hap + i; 
        IF (hap > 1000) THEN 
		LEAVE myWhile; -- 지정한 label문을 떠남. 즉, While 종료.
	END IF;
        SET i = i + 1;
    END WHILE;

    SELECT hap;   
END $$
DELIMITER ;
```

# 에러 처리

기본적인 구조는 다음과 같습니다.

```
DECLARE action HANDLER FOR error_code query
```

action에는 크게 CONTINUE와 EXIT가 사용되며 CONTINUE는 뒤의 query를 실행시키며 EXIT는 넘어가게 됩니다.

- 예시 1

아래의 예시에서 에러코드 1146은 없는 테이블을 조회시 발생되고, 이 때의 action이 CONTINUE이므로 뒤의 query인 SELECT문이 실행됩니다.

```
DELIMITER $$
CREATE PROCEDURE errorProc()
BEGIN
    DECLARE CONTINUE HANDLER FOR 1146 SELECT '테이블이 없어요ㅠㅠ' AS '메시지';
    SELECT * FROM noTable;  -- noTable은 없음.  
END $$
DELIMITER ;
```

- 예시 2

에러코드와 상태코드를 둘 다 사용할 수 있으며 아래의 코드 상 에러가 생기면 BEGIN END; 가 실행됩니다.

```
DELIMITER $$
CREATE PROCEDURE errorProc2()
BEGIN
    DECLARE CONTINUE HANDLER FOR SQLEXCEPTION 
    BEGIN
	SHOW ERRORS; -- 오류 메시지를 보여 준다.
	SELECT '오류가 발생했네요. 작업은 취소시켰습니다.' AS '메시지'; 
	ROLLBACK; -- 오류 발생시 작업을 롤백시킨다.
    END;
    INSERT INTO usertbl VALUES('LSG', '이상구', 1988, '서울', NULL, 
		NULL, 170, CURRENT_DATE()); -- 중복되는 아이디이므로 오류 발생
END $$
DELIMITER ;
```

- 예시 3

```
DELIMITER $$
CREATE PROCEDURE errorProc()
BEGIN
    DECLARE i INT; 
    DECLARE hap INT; 
    -- 오버플로우 직전 최대값
    DECLARE saveHap INT; 

    -- 오버플로우에 대한 에러코드 1264
    -- 그 아래 BEGIN END로 쿼리문을 구성
    DECLARE EXIT HANDLER FOR 1264 
    BEGIN
        SELECT CONCAT('INT 오버플로 직전의 합계 --> ', saveHap);
        SELECT CONCAT('1+2+3+4+...+',i ,'=오버플로');
    END;
    
    SET i = 1; 
    SET hap = 0; 
    
    WHILE (TRUE) DO  
        SET saveHap = hap; 
        SET hap = hap + i; 
        SET i = i + 1; 
    END WHILE;
END $$
DELIMITER ;
```

# 동적 쿼리

PREPARE로 쿼리만 준비해놓고 뒤에 EXECURE가 실행되면 그 때 실제로 시작되는 방식으로 작동됩니다. 

```
PREPARE name FROM (쿼리문);
EXECUTE name;
DEALLOCATE PREPARE name;
```

이런 방식이 도움되는 이유로, 프로시저에 값을 넣어주어야하는 경우 먼저 프로시저를 선언해놓고 변수값이 정해지면 넣어서 실행하도록 할 수 있습니다.
이 때 변수가 들어가는 자리에 ?를 넣어서 사용합니다.

- 예시 1
```
SET @myVar1 =3;
PREPARE myQuery
	FROM 'SELECT Name, height FROM usertbl ORDER BY height LIMIT ?';
EXECUTE myQuery USING @myVar1;

```

- 예시 2: 테이블 이름을 사용하는 경우

기존의 방식으로 할 경우 에러가 생깁니다. 
넣어주는 데이터는 VARCHAR(20)인데 테이블 이름은 기존에도 이 형식으로 사용하지 않았기 때문에 납득이 됩니다.

```
DELIMITER $$
CREATE PROCEDURE nameProc(
    IN tblName VARCHAR(20)
)
BEGIN
 SELECT * FROM tblName;
END $$
DELIMITER ;

CALL nameProc ('userTBL');
```

대신 동적 쿼리를 이용해서 쿼리문을 만들어주는 방식으로 진행할 수 있습니다.

먼저 이름을 INPUT 파라미터로 받고 SET을 통해 쿼리문을 만들고 
```
DELIMITER $$
CREATE PROCEDURE nameProc(
    IN tblName VARCHAR(20)
)
BEGIN
  SET @sqlQuery = CONCAT('SELECT * FROM ', tblName);
  -- 여기서 동적 쿼리가 진행됩니다.
  PREPARE myQuery FROM @sqlQuery;
  EXECUTE myQuery;
  DEALLOCATE PREPARE statement;
END $$
DELIMITER ;
```

# 기타

- stored procedure가 어떤 것이 있는지 확인하는 코드

단 파라미터는 확인되지 않습니다.

```
SELECT routine_name, routine_definition FROM INFORMATION_SCHEMA.ROUTINES
    WHERE routine_schema = table_name AND routine_type = 'PROCEDURE';
```

- stored procedure의 파라미터를 확인하는 코드

```
SELECT parameter_mode, parameter_name, dtd_identifier
	FROM INFORMATION_SCHEMA.PARAMETERS
	WHERE specific_name = procedure_name;
```