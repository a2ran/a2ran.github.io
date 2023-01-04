---
title: "[Lecture Note] MySQL 06. Table Structure"
toc: true
use_math: true
categories:
  - RDBMS
tags:
  - [MySQL]
---

[이것이 MySQL이다 강의](https://www.youtube.com/watch?v=xKYeJxBTt2E&list=PLVsNizTWUw7Hox7NMhenT-bulldCp9HP9)를 참고하여 정리한 글임을 밝힙니다.


# 데이터 관리

## 테이블 단위

### 테이블 삭제

```
DELETE FROM bigTbl1; -- 데이터가 삭제 & 한행식 진행해서 괭장히 느림
DROP TABLE bigTbl2; -- 테이블 자체가 삭제 & 그나마 빠름, 한꺼번에 지운다
TRUNCATE TABLE bigTbl3; -- 테이블의 구조만 남아 DELETE와 동일 & 한꺼번에 지우는 방식으로 동일
```

## 열 단위
### 열의 추가

```
ALTER TABLE usertbl
	ADD homepage VARCHAR(30)  -- 열추가
		DEFAULT 'http://www.hanbit.co.kr' -- 디폴트값
		NULL; -- Null 허용함
```

### 열의 삭제 

```
ALTER TABLE usertbl
	DROP COLUMN mobile1;
```

단 제약조건이 있는 경우 먼저 제약 조건을 지워야 한다.

- 제약조건 지우는 예시

```
ALTER TABLE buytbl
	DROP FOREIGN KEY buytbl_ibfk_1;
```

### 열의 변경


```
ALTER TABLE usertbl
	CHANGE COLUMN name uName VARCHAR(20) NULL ;
```

위의 예시에선 name에서 uName과 나머지 자료형과 제약조건이 서술됩니다.

# 제약조건 
테이블을 생성할 때 추가될 수  있는 조건은 매우 많으며 그 일부만 정리하도록 하겠습니다. 

크게 사용되는 제약조건들은 다음과 같습니다.

- [PRIMARY KEY](##primary-key)

- [FOREIGN KEY](##foreign-key)

- [UNIQUE](#unique)

- [CHECK](#check)

- [DEFAULT](#default-%EC%A0%95%EC%9D%98)

- NULL(NOT NULL)



## Primary Key

중복도, NULL도 아니어야하는 데이터를 PK로 사용할 수 있다.

- 지정방식 1 : CONTRAINT PRIMARY KEY

```
CREATE TABLE tabledb.usertbl (
	userID CHAR(8) NOT NULL,
    name VARCHAR(10) NOT NULL,
    birthYear INT NOT NULL,
    -- 여기 PK_usertbl_userID 가 PK의 이름이 되는 것
    CONSTRAINT PRIMARY KEY PK_usertbl_userID(userID)
    );
```

- 지정방식 2 : ALTER TABLE ~ ADD CONSTRAINT 

```
CREATE TABLE tabledb.usertbl (
	userID CHAR(8) NOT NULL,
    name VARCHAR(10) NOT NULL,
    birthYear INT NOT NULL    
    );
ALTER TABLE usertbl
	ADD CONSTRAINT PK_usertbl_userID
		PRIMARY KEY (userID);
```

- 지정방식 3: 2개의 열을 사용하는 경우

2개의 열을 하나로 사용했을 때 NOT NULL이고 unique한 경우엔 사용할 수 있습니다.

```
CREATE TABLE prodtbl(
	prodCode CHAR(3) NOT NULL,
    prodID CHAR(4) NOT NULL,
    prodDate DATETIME NOT NULL,
    prodCur CHAR(10) NULL    
);
ALTER TABLE prodtbl
	ADD CONSTRAINT PK_prodtbl_prodCode_prodID
		PRIMARY KEY(prodCode, prodID);

```

## FOREIGN KEY

두 테이블 사이의 관계를 선언하여 사용하게 되고, 서로 의존하는 관계가 되게 된다.

이 때 FK가 선언되는 **외래 테이블**은 참조하고 있는 **기준 테이블**에 의존하게 됩니다. 

또한 기준테이블에 사용되는 열은 PRIMARY KEY이거나 UNIQUE 제약조건이 설정되어 있어야 합니다. 

- 지정방식 1 
```
CREATE TABLE tabledb.buytbl(
	num INT AUTO_INCREMENT NOT NULL PRIMARY KEY,
    userid CHAR(8) NOT NULL,
    prodName CHAR(6) NOT NULL,
    FOREIGN KEY(userid) REFERENCES usertbl(userid)
);
```

- 지정방식 2
```
CREATE TABLE tabledb.buytbl(
	num INT AUTO_INCREMENT NOT NULL PRIMARY KEY,
    userid CHAR(8) NOT NULL,
    prodName CHAR(6) NOT NULL,
    CONSTRAINT FK_usertbl_buytbl FOREIGN KEY(userid) REFERENCES usertbl(userid)
);
```


## ON UPDATE/DELETE CASCADE

특히 기준 테이블의 변화를 함께 반영하여 업데이트가 되도록 하는 기능을 수행하고 이를 설정하지 않은 경우 

> ON UPDATE NO ACTION 및 ON DELETE NO ACTION

과 동일한 상태가 됩니다.

- 지정방식

```
ALTER TABLE buytbl
	ADD CONSTRAINT FK_usertbl_buytbl
		FOREIGN KEY (userID)
        REFERENCES usertbl (userID)
		ON UPDATE CASCADE
        ON DELETE CASCADE;
```

## UNIQUE

PK와 유사하지만 NULL값을 허용한다는 차이점이 있습니다. 

- 지정 방식

```
CREATE TABLE usertbl 
( userID  CHAR(8) NOT NULL PRIMARY KEY, 
  -- 여기서
  email   CHAR(30) NULL  UNIQUE
);
```
혹은
```
DROP TABLE IF EXISTS usertbl;
CREATE TABLE usertbl 
( userID  CHAR(8) NOT NULL PRIMARY KEY,
...
  email   CHAR(30) NULL ,  
  -- CONSTRAINT로 추가하고 이름은 주로 AK를 붙인다고 합니다.
  CONSTRAINT AK_email  UNIQUE (email)
);

```

## CHECK

입력시 데이터 값을 검사하는 역할을 합니다. 
- 지정 방식

```
CREATE TABLE usertbl 
( userID  CHAR(8) PRIMARY KEY,
...
  -- WHERE에 사용되는 것처럼 사용하면 되는 것 같습니다 .
  birthYear  INT CHECK  (birthYear >= 1900 AND birthYear <= 2023),
);
```

ALTER TABLE로 지정하는 방식
```
ALTER TABLE usertbl
	ADD CONSTRAINT CK_mobile1
	CHECK  (mobile1 IN ('010','011','016','017','018','019')) ;
```

## DEFAULT 정의

값을 지정하지 않았을 때 자동으로 입력하는 방법입니다.

- 지정방식

```
CREATE TABLE usertbl 
( userID  	CHAR(8) NOT NULL PRIMARY KEY,  
...
  birthYear	INT NOT NULL DEFAULT -1,
  addr	  	CHAR(2) NOT NULL DEFAULT '서울',
  height	SMALLINT NULL DEFAULT 170, 
);
```

ALTER TABLE의 ALTER COLUMN으로 지정하는 방식
```
ALTER TABLE usertbl
	ALTER COLUMN birthYear SET DEFAULT -1;
ALTER TABLE usertbl
	ALTER COLUMN addr SET DEFAULT '서울';
ALTER TABLE usertbl
	ALTER COLUMN height SET DEFAULT 170;
```

- DEFAULT 활용방식

```
-- default임을 알려주는 방식
INSERT INTO usertbl VALUES ('LHL', '이혜리', default, default, '011', '1234567', default, '2023.12.12');
-- 알아서 default 사용하는 방식
INSERT INTO usertbl(userID, name) VALUES('KAY', '김아영');
```

# 에러 유형

- 유형 1 : FK 사용시 기준값 null 문제

아래의 경우 외래 테이블에 데이터를 추가할 때 기준 테이블에 없는 값이라면 에러가 발생하게 됩니다. 

```
ALTER TABLE buytbl
	ADD CONSTRAINT FK_usertbl_buytbl
    FOREIGN KEY (userID)
	REFERENCES usertbl (userID);
```

그래서 일단 무시하고 넣는 경우 아래의 쿼리를 함께 실행합니다.

```
SET foreign_key_checks = 0;
(인서트)
SET foreign_key_checks = 1;
```

- 유형 2 : INSERT 시 에러

에러가 있어서 중간에 멈추는 경우엔 IGNORE란 예약어를 넣어 넘어가게 합니다.

```
INSERT IGNORE INTO membertbl VALUES('BBK', '바비코', '미국'); 
INSERT IGNORE INTO membertbl VALUES('SJH', '서장훈', '서울');
INSERT IGNORE INTO membertbl VALUES('HJY', '현주엽', '경기');
SELECT * FROM membertbl;
```

특히 PK가 이미 있어서 에러가 발생하는 경우 업데이트를 하도록 명령을 내릴 수도 있습니다.

```
INSERT INTO membertbl VALUES('BBK', '바비코', '미국')
	ON DUPLICATE KEY UPDATE name='바비코', addr='미국';
INSERT INTO membertbl VALUES('DJM', '동짜몽', '일본')
	ON DUPLICATE KEY UPDATE name='동짜몽', addr='일본';
```

- 유형 3: FK 사용시 PK 수정하는 경우

앞과 동일하게 FK 확인을 끄고 업데이트 진행 

```
SET foreign_key_checks = 0;
UPDATE usertbl SET userID = 'VVK' WHERE userID='BBK';
SET foreign_key_checks = 1;
```

이런 것들을 방지하려고 ON UPDATE CASCADE를 사용하는 것입니다!

- 유형 3 : FK 사용시 데이터를 삭제하는 경우 

테이블을 지울 땐 FK를 사용하는 테이블을 먼저 지워야합니다. 그렇지 않으면 에러가 발생하게 되고

방법 1 : ON DELETE CASCADE 사용하여 같이 삭제

방법 2 : 연결된 구조를 보고 FK 먼저 지우고 그다음에 PK 지우기

먼저 table_named의 제약조건들을 봅니다. 
```
SELECT table_name, constraint_name
    FROM information_schema.referential_constraints
    WHERE constraint_schema = table_name;
```

그 다음 FK가 있는 테이블에서 FK를 지우고 나서 PK를 지웁니다.
```
ALTER TABLE fk_table_name DROP FOREIGN KEY fk_name;
ALTER TABLE pk_table_name DROP PRIMARY KEY;
```

# 그 외
## 테이블 압축

- 선언 방식

ROW_FORMAT에 COMPRESSED를 추가해서 데이터를 **내부적**으로 압축해서 관리하게 됩니다.<br>
SELECT할 때에는 겉으론 똑같지만 내부적으로 압축과정이 있어 속도가 느려지기도 합니다.

```
CREATE TABLE compressTBL( emp_no int , first_name varchar(14))
	ROW_FORMAT=COMPRESSED ;
```

## 임시 테이블
현재 세션(접속자)에만 적용되며 직접 지우거나 Workbench가 종료 혹은 서버가 재부팅되면 사라지게 됩니다. 

이렇게 사용하는 이유로 작업 중간에 잠깐 쓰는 테이블과 중요 테이블을 구분하기 위함입니다.

- 지정 방식
CREATE TEMPORARY TABLE로 선언하게 됩니다. 이 때 사용된 이름이 기존 테이블과 같다면 기존 테이블은 사용되지 않는다고 합니다. 
```
CREATE TEMPORARY TABLE  IF NOT EXISTS employees (id INT, name CHAR(5));
```
