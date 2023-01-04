---
title: "[Lecture Note] 02 MySQL과 실습"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - PHP
tags:
  - [PHP, MySQL]
---

[이것이 MySQL이다 강의](https://www.youtube.com/watch?v=xKYeJxBTt2E&list=PLVsNizTWUw7Hox7NMhenT-bulldCp9HP9)를 참고하여 정리한 글임을 밝힙니다.


# 시스템 구성

- 화면 조회
- 신규 회원
- 회원 정보 수정
- 회원 삭제 

# 화면 조회

직접 HTML 코드를 작성하는 방식으로 테이블을 만들 수 있습니다.

또한 다른 페이지로 라우팅할 때 get method로 정보를 가져가기 위해 a 태그의 href에 userID를 가져가는 기능도 포함되어 있습니다.

```
$sql ="SELECT * FROM userTBL";
$ret = mysqli_query($con, $sql);   
...

echo "<h1> 회원 조회 결과 </h1>";
echo "<TABLE border=1>";
echo "<TR>";
...
echo "</TR>";

while($row = mysqli_fetch_array($ret)) {
echo "<TR>";
echo "<TD>", $row['userID'], "</TD>";
...
echo "<TD>", "<a href='update.php?userID=", $row['userID'], "'>수정</a></TD>";
...
echo "</TR>";	  
}   
```

# 신규 회원

INSERT할 때 FORM의 method가 post였기 때문에 $_post로 값을 받게 됩니다. 

```
$userID = $_POST["userID"];
...
$mDate = date("Y-m-j");

$sql =" INSERT INTO userTbl VALUES('".$userID."','".$name."',".$birthYear;
$sql = $sql.",'".$addr."','".$mobile1."','".$mobile2."',".$height.",'".$mDate."')";

$ret = mysqli_query($con, $sql);
```

# 회원 정보 수정

우선 입력받은 ID에 따라서 WHERE문으로 쿼리를 작성합니다.

```
$sql ="SELECT * FROM userTBL WHERE userID='".$_GET['userID']."'";
```

중간에 PHP 코드를 실행하도록 하여 HTML을 작성할 수 있습니다.<br>
또한 INPUT 태그 중 읽기 전용은 READONLY를 추가하여 수정을 불가능하게 합니다.

```
아이디 : <INPUT TYPE ="text" NAME="userID" VALUE=<?php echo $userID ?> READONLY> <br>
...
<INPUT TYPE="submit"  VALUE="정보 수정">
```

# 회원 삭제 

위의 내용으로 구현이 가능하게 됩니다.