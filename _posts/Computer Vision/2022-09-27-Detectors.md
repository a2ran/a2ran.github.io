---
title: "[Lecture] Detectors: Computer Vision for Data Science"
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

how can we align images?
- 단순히 image의 평행이동이 아니라, target에 대한 rotation, translation의 경우!
- 꼭 이미

1. corner & feature를 찾기
  - 결국 어떻게 **detect**하는지?
  - Conrner & Edge!
2. matching
  - detect한 그 feature를 어떻게 describe하는지?


# Edge Detection

- Edge는 이미지상에서 어떤 것인가?
  - large pixel change or "**Discontinuity**"
  - ex. Depth, Distance, orientation, color, 

## Gradient of Image

결국 이미지의 discontinuity를 확인하기엔 gradient가 적절!<br>
다만, 이미지의 gradient의 경우 noise의 variance가 2배가 된다.

![제목](/assets/images/Computer Vision/7-1.PNG){: width="70%" height="55%"}{: .align-center}

여기서 말하고 있는 것이, 우리가 궁금해하는 target의 image($f$)를 바로 미분해서 결과를 본다면, noise가 더 심해진다는 것!

![제목](/assets/images/Computer Vision/7-2.PNG){: width="70%" height="55%"}{: .align-center}

## Gradient of Filter

그래서 gaussian filter는 결국 smoothing효과가 있었으니, gaussian filter를 사용한 결과를 미분해보자!

![제목](/assets/images/Computer Vision/7-3.PNG){: width="70%" height="55%"}{: .align-center}

filter를 사용한 이미지의 gradient는 kernel of filter를 미분한 것을 filtering하는 것과 동일한 효과

![제목](/assets/images/Computer Vision/7-4.PNG){: width="70%" height="55%"}{: .align-center}

그 결과, 우린 **gaussian derivative filter를 edge detection에** 사용하게 된다!

# Corner Detection

## Properties

- Repeatable: invariant w.r.t distortion(shift, brightness and etc)
- Saliency: be distinctive
- Compactness: except trivial result(ex. every pixel)
- Locality: depend on the local part of total image

이 또한 결국 **intensity의 차이**가 중요한 특징이다!
- corner도 결국 일종의 edge이니 intensity 차이를 보는 것
- 그렇지만 edge는 그냥 gradient가 충분히 크기만 하면 되는 것! 그래서 단순히 gaussian derivative filter를 사용
- 다만 corner는 그 **방향성**도 중요하다! 

## Formulation

### Energy function

![제목](/assets/images/Computer Vision/7-5.PNG){: width="70%" height="55%"}{: .align-center}

다만 위의 식은 reference인 ($(u,v)$)에 대해서도, 또한 window size 내의 모든 점에 대해서도 계산하려면 complexity 문제가 있을 것.

### Tayer expansion

![제목](/assets/images/Computer Vision/7-6.PNG){: width="70%" height="55%"}{: .align-center}

### Quadratic Form

![제목](/assets/images/Computer Vision/7-7.PNG){: width="70%" height="55%"}{: .align-center}

이렇게 quadratic form에서 off-diagonal은 0이라고 가정했을 때,<br>
corner가 존재한다는 것은 diagonal 값은 큰 경우!

- Contour

또한 지금 2 by 2 matrix에서 off-diagonal은 0이라고 가정했으니 symmetric하게 되고, 그 결과 contour는 ellipse 형태가 된다.

- Spectral Decomposition

이런 Second Moment Matrix에 spectral decomposition을 취하게 되면, scaling as eigenvalues & rotating as eigenvectors!
- 특히 여기 matrix spectrum을 eigenvalue의 역수로 정렬하기도 하는데, 이는 plotting을 위한 것!

### Eigenvalue of 2nd Moment matrix

결국 Energy function의 2nd moment matrix의 2개의 eigenvalue 모두 크고, 그 둘이 비슷비슷한 크기를 가지는 경우에는 corner로 detection하게 된다!<br>
이 때 eigenvalue를 바로 계산, 비교하는 것은 어렵기 때문에 이에 대한 auxiliary로 metric $R$을 이용한다.

![제목](/assets/images/Computer Vision/7-8.PNG){: width="70%" height="55%"}{: .align-center}

## Others: Harris Corner Detector

앞에서 사용한 2nc Moment matrix에 weight를 추가한 형태를 사용하는 것!

![제목](/assets/images/Computer Vision/7-9.PNG){: width="70%" height="55%"}{: .align-center}

다만 이렇게 계산한 R에 대해서 
- threshold를 정해줘야 하며
- R 값의 local maxima를 찾아야 한다.
  - NMS: Non-mamxima Suppression

![제목](/assets/images/Computer Vision/7-10.PNG){: width="70%" height="55%"}{: .align-center}

## Back to the properties

이렇게 detection을 위한 방식들을 찾았는데, 처음에 얘기한 properties중 결국 repeatable이 중요하다!

즉 우리가 이미지에 대해서 여러 transform을 사용하는데, 이 때 지금 고안한 detection들에 대해서 repeatable한 transform을 정리해보자!

![제목](/assets/images/Computer Vision/7-11.PNG){: width="70%" height="55%"}{: .align-center}

### Affine transform

여기서 affine은 intensity에 대한 것으로, 기존의 intensity 값을 변형한다는 것인데 edge든 corner든 intensity change(=gradient)만 보는 상황에서 translation은 문제가 없고, 다만 coefficient는 threshold도 변형해주어야 한다는 점에서 partially invariant!

![제목](/assets/images/Computer Vision/7-12.PNG){: width="70%" height="55%"}{: .align-center}

### Image Translation

translation은 이미 convolution에 대해 equivariant하다고 알려져 있고, edge 또한 convolution을 이용하고 있으며 translation도 linear transformation이니 spectral decomposition을 이용한 corner에 대해서도 invariant하게 나올 것!

### Image rotation

rotation도 linear하고 결국 convolution으로 본다면 edge에도 무관할 것.

또한 corner detection에서 사용하는 것은 eigenvalue이고 이건 scaling factor이니 rotation과 무관

### Image Scaling

다만 scaling에는 당연히 variant하다...!!!
