---
title: "[Lecture Note] MySQL 01. Overall"
toc: true
use_math: true
categories:
  - RDBMS
tags:
  - [MySQL]
---
[이것이 MySQL이다 강의](https://www.youtube.com/watch?v=xKYeJxBTt2E&list=PLVsNizTWUw7Hox7NMhenT-bulldCp9HP9)를 참고하여 정리한 글임을 밝힙니다.

# Introduction

**DB의 특징**

- 데이터 무결성: 오류 없음
- 데이터 독립성: 응용 프로그램과 무관히 운영됩니다.
- 그 외 : 보안, 중복의 최소화 

**DB의 진화 과정** 
수직적 구조인 계층형 -> 수직수평 구조인 망형 -> 관계형

이제 다룰 관계형 데이터베이스(RDB)의 핵심은 **Table**(Relation, entity) 형태로 자료를 사용하는 것

관계형 데이터 베이스를 관리하기 위한 데이터 베이스 운영 소프트웨어(RDBMS)에서 사용하는 언어가 SQL(*Structured Query Language*)입니다.


# 자료 구조

전체 데이터 베이스(혹은 Schema) 아래에 여러 개의 테이블로 구성됩니다.

테이블은 크게 행(row)와 열(column)으로 구성이 되며

- 행은 최소 데이터 집합이며 행의 크기는 곧 데이터의 크기가 됩니다.
- 열은 데이터 타입이 정해져 있으며 각 데이터의 이름이 붙여져 있습니다.

# Modeling

실제 세계에 있는 데이터를 테이블로 구현하는 방법을 말하며 

**정규화**, **비정규화**와 같은 실무적인 내용은 추후에 다루기로 하며 개념만 다루도록 합니다.

간단히 전체 프로세스를 정리하면 다음과 같습니다.

1.  실제 데이터를 정리합니다.
 - null data가 있는 row를 먼저 위로 올려 L자형 테이블로 구성합니다.
2. L자형 테이블을 보고 데이터를 구분해봅니다.
 - 이렇게 구분할 땐 공통적으로 가지는 속성이 무엇인지 보아야 합니다.
 - 이 때 보통 1:N의 관계를 가지게 되며 주로 **PK**(Primary Key) : **FK**(Foreign Key) 의 관계가 됩니다. 
3. 앞서 나눈 테이블에서 각 column의 데이터 타입과 Null 여부, PK/FK를 구분하여 정리합니다. 
