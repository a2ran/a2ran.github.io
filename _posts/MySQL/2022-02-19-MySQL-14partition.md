---
title: "[Lecture Note] MySQL 14. Partition"
toc: true
toc_sticky: true
use_math: true
categories:
  - RDBMS
tags:
  - [MySQL]
---

[이것이 MySQL이다 강의](https://www.youtube.com/watch?v=xKYeJxBTt2E&list=PLVsNizTWUw7Hox7NMhenT-bulldCp9HP9)를 참고하여 정리한 글임을 밝힙니다.


# Introduction 

대량의 테이블을 **내부적**으로 나누는 작업을 **parititon**이라고 합니다.

분할 기준은 연도별, 월별 등의 기준을 정해놓아야 하며 쿼리를 어떻게 할 것이며 업무를 어떻게 분할할 것인지에 따라 주의하여 정해야 합니다.

# Declaration

나누는 기준값들을 **파티션 키**라고 부릅니다.

- 레인지 파티션(범위로 나눌 때)

연속된 숫자형의 범위로 구분합니다.

```
PARTITION BY RANGE(birthYear) (
    PARTITION part1 VALUES LESS THAN (1971),
    PARTITION part2 VALUES LESS THAN (1979),
    PARTITION part3 VALUES LESS THAN MAXVALUE
);
```

- 리스트 파티션(카테고리로 나눌 때)

파티션 키 값들을 직접 지정해야 하며  앞서 사용한 MAXVALUE는 리스트 파티션에선 사용할 수 없습니다.

```
PARTITION BY LIST_COLUMNS(addr) (
    PARTITION part1 VALUES VALUES_IN ('~', '~'),
    ...
);
```



# Implementation

주어진 쿼리에 따라 파티션별로 작업을 하게 됩니다. 그래서 단순 전체 읽기를 한 경우 데이터가 파티션별로 모여 있는 것을 확인할 수 있습니다.

- 예시 : 수정(분할)

```
ALTER TABLE partTBL 
	REORGANIZE PARTITION part3 INTO (
		PARTITION part3 VALUES LESS THAN (1986),
		PARTITION part4 VALUES LESS THAN MAXVALUE
	);

-- 분할 적용
OPTIMIZE TABLE partTBL;
```

- 예시 : 수정(병합)

```
ALTER TABLE partTBL 
	REORGANIZE PARTITION part1, part2 INTO (
		PARTITION part12 VALUES LESS THAN (1979)
	);

-- 병합 적용
OPTIMIZE TABLE partTBL;
```

- 예시 제거

파티션을 지우면 데이터도 사라지니 주의해야 합니다.

```
ALTER TABLE partTBL DROP PARTITION part12;
OPTIMIZE TABLE partTBL;
```

# 특징

- stored procedure, stored function, 사용자 변수 등 파티션 함수나 식에 사용 불가능 합니다.
- 파티션 키에는 일부 함수만 사용가능 합니다.

# 기타

- 파티션 확인

```
SELECT TABLE_SCHEMA, TABLE_NAME, PARTITION_NAME, PARTITION_ORDINAL_POSITION, TABLE_ROWS
    FROM INFORMATION_SCHEMA.PARTITIONS
    WHERE TABLE_NAME = table_name;
```

- 쿼리 수행 과정 설명 쿼리

```
EXPLAIN SELECT ~;
```