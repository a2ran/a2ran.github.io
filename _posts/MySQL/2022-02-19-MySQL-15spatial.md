---
title: "[Lecture Note] MySQL 15. Spatial Data"
toc: true
toc_sticky: true
use_math: true
categories:
  - RDBMS
tags:
  - [MySQL]
---

[이것이 MySQL이다 강의](https://www.youtube.com/watch?v=xKYeJxBTt2E&list=PLVsNizTWUw7Hox7NMhenT-bulldCp9HP9)를 참고하여 정리한 글임을 밝힙니다.



지리 정보 시스템은 지도를 포함해서, 네비게이션같은 서비스에 사용되는 정보를 말합니다.

공간데이터를 파일로 따로 존재하고 DMBS와 함께 사용하여 테이블의 속성을 조합하면 GIS 응용 프로그램이 되었습니다.

그리고 이젠 공간데이터를 MySQL에 저장할 수 있는 데이터타입(Geometry)을 지원하게 되었습니다.

# 공간데이터의 구성

- 점(POINT) : 1개의 x,y 좌표
- 선(LINESTRING) : 여러 개의 x,y 좌표
- 면(POLYGON) : 여러 개의 x,y좌표 & 첫 점 = 끝 점

# MySQL에서 적용

5.0이전엔 그냥 LONGBLOB으로 파일 자체를 저장, 사용할 때엔 다른 툴로 다시 열어서 사용해야 합니다.<br>
그 이후론 Geometry type으로 개체(entity)단위로 데이터를 저장할 수 있게 되어서 DB 안에서 조회 및 처리가 가능하게 됩니다. 또한 Spatial Viewer를 지원하여서 결과를 볼 수 있습니다.

# Geometry 함수

기존의 함수들은 적용이 안되고 정해진 함수를 사용해야 데이터 처리가 가능하게 됩니다.

ex.
- ST_GeomFromText()
- ST.Intersects()
...



# Implementation

- 테이블 생성
```
CREATE TABLE StreamTbl (
   MapNumber CHAR(10),  
   StreamName CHAR(20), 
   -- 공간데이터이니 GEOMETRY로 선언
   Stream GEOMETRY ); 
```

- 데이터 입력

LINESTRING의 경우 x,y 좌표가 쌍으로 하나씩 추가된 형태로 입력됩니다.
```
INSERT INTO StreamTbl VALUES ( '330000001' ,  '한류천', 
	ST_GeomFromText('LINESTRING (-10 30, -50 70, 50 70)'));
```

POLYGON의 경우엔 괄호가 하나 더 들어가는 형태가 되어야 하며 이 때 시작하는 쌍과 끝 쌍은 같은 값을 가져야 합니다.
```
INSERT INTO BuildingTbl VALUES ('330000005' ,  '하나은행', 
	ST_GeomFromText('POLYGON ((-10 50, 10 30, -10 10, -30 30, -10 50))'));
```

- 쿼리
데이터에 실제 140이란 정보는 없지만 공간 데이터를 주었을 때 강 길이를 알아서 계산해서 이런 쿼리가 가능하게 됩니다.

```
SELECT * FROM StreamTbl WHERE ST_Length(Stream) > 140 ;
```

면적도 알아서 계산하게 됩니다.
```
SELECT BuildingName, ST_AREA(Building) FROM BuildingTbl 
	WHERE ST_AREA(Building) < 500;
```

- 겹치는 대상 조회

아래의 경우 안양천과 만나는 건물들을 조회하게 됩니다.
```
SELECT StreamName, BuildingName, Building, Stream
   FROM BuildingTbl , StreamTbl 
   WHERE ST_Intersects(Building, Stream) = 1   AND StreamName = '안양천';
```

- 선의 두께 조정

Buffer로 두껍게 표시할 수 있습니다.
```
SELECT ST_Buffer(Stream,5) FROM StreamTbl;
```

- 두 지점 사이 거리 표시

동일한 테이블을 다른 이름(R1, R2)로 표시해서 ST_Distance를 사용하게 되면 두 점 사이 거리를 구할 수 있게 됩니다.

```
SELECT R2.restName,
       R2.restAddr,
       R2.restPhone, 
       ST_Distance(R1.restLocation, R2.restLocation) AS "1호점에서 거리"
FROM Restaurant R1, Restaurant R2
WHERE R1.restName='왕매워 짬뽕 1호점'
ORDER BY ST_Distance(R1.restLocation, R2.restLocation) ;
```

- 겹치는 지점 필터링

```
-- 각 지역을 변수에 할당
SELECT  Area INTO @eastNorthSeoul FROM Manager WHERE ManagerName = '존밴이';
SELECT  Area INTO @westSeoul FROM Manager WHERE ManagerName = '당탕이';
-- Intersection을 찾은 다음에 해당 열
SELECT  ST_Intersection(@eastNorthSeoul, @westSeoul) INTO @crossArea ;
-- Intersection에 포함여부를 WHERE로
SELECT DISTINCT R.restName AS "중복 관리 지점"
    FROM Restaurant R, Manager M
    WHERE ST_Contains(@crossArea, R.restLocation) = 1;
```