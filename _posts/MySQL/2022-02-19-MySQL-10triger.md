---
title: "[Lecture Note] MySQL 10. Triger"
toc: true
use_math: true
categories:
  - RDBMS
tags:
  - [MySQL]
---

[이것이 MySQL이다 강의](https://www.youtube.com/watch?v=xKYeJxBTt2E&list=PLVsNizTWUw7Hox7NMhenT-bulldCp9HP9)를 참고하여 정리한 글임을 밝힙니다.


특정 테이블에 할당되어 DML문(**INSERT, UPDATE, DELETE**같은 작업들이며 **TRUNCATE의 경우 DDL문**이라서 트리거가 되지 않습니다)이 발생되면 자동으로 실행되는 프로그래밍이라고 생각하면 될 것 같습니다.

트리거의 특성상 강제로 사용할 수 없고 이벤트가 발생해야 하며 파라미터도받을 수 없습니다.

또한 ALTER를 사용못하므로 변경시 지우고 생성해야 합니다.

# 기본 형태

```
DELIMITER // 
CREATE TRIGGER triger_name
    (AFTER| BEFORE) (INSERT | DELETE | UPDATE)
    ON target_table
    -- 그냥 무조건 쓰는 것, 각 행마다 작업하게 되는 것
    FOR EACH ROW 
    -- optional로 트리거의 순서를 결정
    (FOLLOW | PRECEDES) other_trigger_name
BEGIN
	(src)
END // 
DELIMITER ;
```

# 임시 테이블

트리거가 발생하면 트리거 이전 혹은 바뀔 데이터를 다룰 수 있게 되고 이때 사용되는 예약어가 OLD와 NEW입니다.

BEFORE 트리거의 경우에는 바뀌기 전이니 NEW가 사용되고, AFTER 트리거의 경우에는 바뀌고 난 후니 OLD가 사용됩니다.

주로 INSERT의 경우 BEFORE로 사용되어 NEW를 사용하고 DELETE는 AFTER로 사용되어 OLD를 사용하고 UPDATE는 둘 다 사용될 수 있게 됩니다.


# 여러 트리거(다중, 중첩 트리거)

- 다중 트리거

 하나의 이벤트에 여러 트리거를 사용하는 것입니다. 이 경우 작동 순서를 지정할 수 있습니다.

- 중첩트리거

 하나의 트리거가 다른 트리거를 작동하게 만드는 트리거 입니다. 만약 열 이름이 바뀐다면 업데이트가 안되어서 에러가 생길 수 있습니다.



- 예시 1

```
DELIMITER //
CREATE TRIGGER trg_deletedMembertbl	AFTER DELETE ON membertbl FOR EACH ROW
BEGIN	
INSERT INTO deletedMemberTBL VALUES (OLD.memberID, OLD.memberName, OLD.memberAddress, CURDATE() );
END //
DELIMITER ;
```

이렇게 TRIGER를 작성하게 된다면 
- 이름: trg_deletedMebertbl
- 실행 조건 : DELETE 문
- 실행 내용 : INSERT문

으로 정리할 수 있게 됩니다.

위의 쿼리문으로 본다면 membertbl에서 DELETE문이 실행된 경우 지워지는 데이터를 deleteMemberTBL이란 테이블에 입력하게 됩니다.

이 때 OLD는 예약어로, TRIGGER가 시작하게 만든 data를 가리키고 있습니다.


- 예시 2 : UPDATE 트리거 

```
DELIMITER $$
CREATE TRIGGER backUserTbl_UpdateTrg  
    AFTER UPDATE 
    ON userTBL 
    FOR EACH ROW 
BEGIN
    INSERT INTO backup_userTbl VALUES( OLD.userID, OLD.name, OLD.birthYear, 
        OLD.addr, OLD.mobile1, OLD.mobile2, OLD.height, OLD.mDate, 
        '수정', CURDATE(), CURRENT_USER() );
END $$
DELIMITER ;
```

- 예시 3 : DELETE 트리거

```
DELIMITER // 
CREATE TRIGGER backUserTbl_DeleteTrg  
    AFTER DELETE 
    ON userTBL 
    FOR EACH ROW 
BEGIN
    INSERT INTO backup_userTbl VALUES( OLD.userID, OLD.name, OLD.birthYear, 
        OLD.addr, OLD.mobile1, OLD.mobile2, OLD.height, OLD.mDate, 
        '삭제', CURDATE(), CURRENT_USER() );
END // 
DELIMITER ;
```

- 예시 4 : 의도적 에러 발생 트리거

SIGNAL SQLSTATE을 사용하면 실제 작동은 ROLLBACK 되어서 작동하지 않게 됩니다.

```
DELIMITER // 
CREATE TRIGGER userTbl_InsertTrg  
    AFTER INSERT 
    ON userTBL 
    FOR EACH ROW 
BEGIN
    SIGNAL SQLSTATE '45000' 
        SET MESSAGE_TEXT = '데이터의 입력을 시도했습니다. 귀하의 정보가 서버에 기록되었습니다.';
END // 
DELIMITER ;
```

- 예시 5 : BEFORE 트리거와 NEW

일종에 값을 실행하기전에 검사하는 역할로 사용할 수 있습니다.

```
DELIMITER // 
CREATE TRIGGER userTbl_BeforeInsertTrg  
    BEFORE INSERT 
    ON userTBL 
    FOR EACH ROW 
BEGIN
    IF NEW.birthYear < 1900 THEN
        SET NEW.birthYear = 0;
    ELSEIF NEW.birthYear > YEAR(CURDATE()) THEN
        SET NEW.birthYear = YEAR(CURDATE());
    END IF;
END // 
DELIMITER ;
```

- 예시 6 : 중첩 트리거 

트리거 1 
```
DELIMITER // 
CREATE TRIGGER orderTrg  
    AFTER  INSERT 
    ON orderTBL 
    FOR EACH ROW 
BEGIN
    -- 여기서의 업데이트가 실행되고
    UPDATE prodTbl SET account = account - NEW.orderamount 
        WHERE prodName = NEW.prodName ;
END // 
DELIMITER ;
```

트리거 2
```
DELIMITER // 
CREATE TRIGGER prodTrg  
    --여기서 업데이트를 감지하고 실행되는 것입니다.
    AFTER  UPDATE 
    ON prodTBL 
    FOR EACH ROW 
BEGIN
    DECLARE orderAmount INT;
    -- update이니 NEW와 OLD를 사용할 수 있습니다.    
    SET orderAmount = OLD.account - NEW.account;
    INSERT INTO deliverTbl(prodName, account)
        VALUES(NEW.prodName, orderAmount);
END // 
DELIMITER ;
```

# 기타

트리거 목록을 확인하는 코드

```
SHOW TRIGGERS FROM table_name;
```