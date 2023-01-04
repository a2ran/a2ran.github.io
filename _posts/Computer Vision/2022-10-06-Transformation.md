---
title: "[Lecture] Transformation: Computer Vision for Data Science"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Computer Vision
tags:
  - [Computer Vision]
---

# Motivation

지금까지 사진 내에서 관심 영역을 찾아내고(corner, edge, blob), 관심 영역들 각각 설명하여(histogram of orientation) matching 알고리즘(SIFT)를 구현하는 과정을 다뤘다.

이젠 이렇게 matching된 관심 영역들을 맞춰주는 작업인 transformation을 하려고 한다.

transformation은 parameterization을 통해 model로 fitting이 가능할 것! 다만 문제가 되는 부분과 이에 대한 해결책들은 다음과 같다.

- noise: total least square --> Objective Function!
- outlier: RANSAC --> Data!
- multiple model: Hough Transform

# Problem Definition

## Data

우리가 지금 다루는 데이터는 SIFT 알고리즘으로 계산된 대상들이다(다만, matching된 pair인 것인지 단순히 feature들인 것인지는 불확실, TBD)

이런 데이터들에도 Outlier가 존재하는데 모든 데이터를 쓰려고 하니 생긴 문제다!<br>
outlier를 사용하면 일단 에러가 큰 경우가 존재하니, **에러가 적은 것들 중**에 가능한 많은 데이터를 쓰는 방식도 존재한다!!(이건 에러)

### RANSAC

1. sampling으로 subset을 만들기
2. subset으로 fitting
3. threshold를 기준으로 inlier labelling
4. fitting으로 했을 때 inlier가 많으면 그걸 사용하기!

hyperparameter : 
- The number of trials
- subset size
- threshold of inlier

## Model

How to parameterize it?

### Hough Transform

현재까진 data space상에서 fitting을 하고 있었는데 이건 그 dataset에 가장 적절한 parameter를 구한다는 것이고, 이는 parameter space상에서 가장 적절한 점을 찾는 다는 것과 동일하다.

[제목](/assets/images/Computer Vision/9-8.PNG){: width="70%" height="55%"}{: .align-center}

이를 반대로 생각해보면, 각 데이터 별로 parameter space에서 line을 그릴 수 있고, 그런 line들을 data마다 그려서 intersection을 찾는 것과 동일하다!

[제목](/assets/images/Computer Vision/9-9.PNG){: width="70%" height="55%"}{: .align-center}

다만 generlization을 위해서 parameter space를 polar cooridnate로 나타내고, parameter space를 discretize하여서 가장 적절한 point를 찾는 것!<br>
(이건 마치 histogram으로 가장 좋은 parameter 찾는 것과 동일하네)

[제목](/assets/images/Computer Vision/9-10.PNG){: width="70%" height="55%"}{: .align-center}

## Objective function

How to fit?

이는 사실 fitting의 의미가 error를 어떻게 줄이는가에서 오고, 이는 noise에서 비롯되기 때문에 결국 noise를 해결하는 방법에 대한 것

### Total Least Squares

단순한 Least Square는 y의 계수가 0인 경우에 대해선 문제가 되기도 한다. 그래서 이를 일반화할 필요가 존재한다.

즉, $ax+by+c=\mathbf{l^Tp}=0$ 으로 나타내는 것은 직선의 normal vector $\mathbf{l}$ 과 직선 위의 점 $\mathbf{p}$ 로 나타내는 것이 되고 normal vector는 normalize하여 제한 조건을 둔다면 

[제목](/assets/images/Computer Vision/9-1.PNG){: width="70%" height="55%"}{: .align-center}

여기서 Least square과 동일하게 distance metric을 정의해서 최소화 하자면

[제목](/assets/images/Computer Vision/9-2.PNG){: width="70%" height="55%"}{: .align-center}

결과적으로 다음과 같은 모델링을 얻게 되고 이러한 특징은 rotation에 invariant하다(직선 방향에 무관하게 normal vector를 정의하니!)

[제목](/assets/images/Computer Vision/9-3.PNG){: width="70%" height="55%"}{: .align-center}

이를 해석적으로 푸는 것은 lecture note 참고!!

[제목](/assets/images/Computer Vision/9-4.PNG){: width="70%" height="55%"}{: .align-center}

그 결과 얻은 objective function은 

[제목](/assets/images/Computer Vision/9-5.PNG){: width="70%" height="55%"}{: .align-center}

이러한 구조는 사실 Homogeneous Least Square를 푸는 구조와 동일하다(norm condition도 있고 사실 Spectral Decomposition으로 생각해보면 되는 문제)

[제목](/assets/images/Computer Vision/9-6.PNG){: width="70%" height="55%"}{: .align-center}


다만 이렇게 하면 결국 quadratic이니 error가 크지만 않으면 수렴하는 형태가 되니 해결하고자 해서 다음과 같이 사용하기도 한다.

[제목](/assets/images/Computer Vision/9-7.PNG){: width="70%" height="55%"}{: .align-center}



