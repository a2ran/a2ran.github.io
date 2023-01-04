---
title: "[Lecture Note] MySQL 13. Full Text"
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

어떤 글 내에 검색을 진행할 때 WHERE문을 사용한다고 하면 적절히 수행되지 않기도 합니다.<br> 
```
WHERE col_reference LIKE %key_word%
```
<br>
과 같은 방식으로도 한계가 있습니다. 이런 배경에서 전체를 검색하는 full scan을 한다면 속도가 매우 느려질 것입니다. 

그렇기에 전체 텍스트 검색이란, 엄청나게 긴 글자들을 단어들에 대해 검색할 때 **인덱스**를 사용할 수 있게 합니다.


# FULLTEXT Index

전체 텍스트에 대해서 인덱스를 생성하는 것을 말합니다. 기존의 인덱스로 생각하면 되며 Index_type이 BTREE가 아닌 FULLTEXT로 나오는 것을 확인할 수 있습니다.

특징은 다음과 같습니다.

- InnoDB와 MyISAM에만 적용가능합니다. (우리는 현재 InnoDB를 사용합니다.)
- char, varchar, text에만 적용
- 여러 열에 적용 가능

# Declaration

- 예시 1 

```
CREATE TABLE table_name (col_reference col_data_type, ..., FULLTEXT index_name(col_name));
```

- 예시 2
```
CREATE TABLE table_name (col_reference col_data_type, ..);
ALTER TABLE table_name ADD FULLTEXT index_name(col_name);
```

- 예시 3

```
CREATE TABLE table_name (col_reference col_data_type, ...)
CREATE FULLTEXT INDEX index_name ON col_name;
```

- 예시 4: 삭제

```
ALTER TABLE table_name DROP INDEX FULLTEXT(col_reference);
```

# Stop word

전체 텍스트 인덱스로 사용하지 않을 단어들을 말합니다.

- default

INFORMATION_SCHEMA.INNODB_FT_DEFAULT_STOPWORD 테이블에 36개 정도 가지고 있습니다.

- 생성 후

아래의 쿼리로 확인할 수 있습니다.

```
SELECT word, doc_count, doc_id, position 
	FROM INFORMATION_SCHEMA.INNODB_FT_INDEX_TABLE;
```

- 중지 단어 지정

사전에 중지단어 테이블을 생성해야 하며 이 때 열 이름은 value로 해야 합니다.

```
CREATE TABLE user_stopword (value VARCHAR(30));

-- 이름은 모두 소문자로 사용해야 합니다.
SET GLOBAL innodb_ft_server_stopword_table = 'fulltextdb/user_stopword';
```

# Query

MATCH()와 AGAINST()를 사용합니다.

```
MATCH(col_reference) AGAINTS (expr [search_modifier])

search_modifier:
    {
        IN NATURAL LANGUAGE MODE | IN NATURAL LANGUAGE MODE WITH QUERY EXPANSION |
        iIN BOOLEAN MODE | WITH QUERY EXPANSION
    }
```

## 자연어 검색

기본 상태로는 IN NATURAL LANGUAGE MODE로 자연어 검색을 수행하게 됩니다. 

- 예시 1

```
SELECT * FROM WHERE MATCH(article) AGAINT('영화');
```

특징은 다음과 같습니다.
- '영화가'처럼 조사가 붙어 변형이 있으면 검색이 안됩니다.
- 2가지 키워드로 검색할 경우 띄어쓰기로 같이 AGAINT에 사용하면 됩니다.


## BOOLEAN MODE 

정확히 일치하지 않는 것도 검색해줍니다.

- 예시 1

```
SELECT * FROM WHERE MATCH(article) AGAINT('영화*' IN BOOLEAN MODE);
```

이 경우 '영화가' '영화를'처럼 영화가 포함된 것들을 다 검색하게 됩니다.

- 예시 2

정확히 띄어쓰기가 포함된 키워드를 검색할 경우 BOOLEAN MODE로 검색하면 됩니다.

```
SELECT * FROM WHERE MATCH(article) AGAINT('영화 배우' IN BOOLEAN MODE);
```

- 예시 3 : +

특정 키워드를 포함하는 것만 필터링 하는 경우 다음과 같습니다.

```
SELECT * FROM WHERE MATCH(article) AGAINT('영화+공포' IN BOOLEAN MODE);
```

이렇게 모두 검색되도록 하는 건 두 키워드 앞에 모두 +를 붙이면 됩니다.

```
SELECT * FROM FulltextTbl 
	WHERE MATCH(description) AGAINST('+남자* +여자*' IN BOOLEAN MODE);
```

- 예시 4 : -

특정 키워드를 제외하는 경우 다음과 같습니다.

```
SELECT * FROM WHERE MATCH(article) AGAINT('영화-남자' IN BOOLEAN MODE);
```

- 예시 5

FULLTEXT로 검색한 결과를 SELECT로 출력해보면 일종의 **점수**를 확인할 수 있습니다.

이 값의 의미는 해당 열에서 검색한 키워드 종류가 얼마나 등장하였는지 알려주는 것으로 한 키워드가 몇번 등장하였는지는 무관합니다.

```
SELECT *, MATCH(description) AGAINST('남자* 여자*' IN BOOLEAN MODE) AS 점수 
	FROM FulltextTbl WHERE MATCH(description) AGAINST('남자* 여자*' IN BOOLEAN MODE);
```

# 기타

- 최소 검색 단위 확인 방법

```
SHOW VARIABLES LIKE 'innodb_ft_min_token_size';
```

변경할 경우 my.ini 파일에서 수정 후 서버를 재부팅해야 합니다.







