---
title: "[Lecture] Light-shading: Computer Vision for Data Science"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Computer Vision
tags:
  - [Computer Vision]
---

# Light Theory

check the lecture note!

# In python

- [y, x, c]
차원은 대게 다음과 같은 구조를 가진다

그리고 다음과 같은 특징을 기억하자!

![제목](/assets/images/Computer Vision/3-1.PNG){: width="70%" height="55%"}{: .align-center}

# Light system

## RGB

(R,G,B)로 구성할 땐 간단하지만 distance metric이 정의되어 있지 않다!

![제목](/assets/images/Computer Vision/3-2.PNG){: width="70%" height="55%"}{: .align-center}

## HSV
(H,S,V)로 구성할 땐 각 축의 의미가 명확하고 변환이 쉽지만 이게 더 좋다고는 말못한다.

![제목](/assets/images/Computer Vision/3-3.PNG){: width="70%" height="55%"}{: .align-center}

# Opaque Reflection

우리가 어떤 물체를 본다는 것이 결국 반사된 빛을 본다는 것!

그렇기에 이를 모델링한 것이 바로 <br>
"Bi-directional reflectance distribution function"(**BRDF**)

즉 입사각과 반사각이 어떻게 될 것인지를 내부에서 일어나는 transmittion, reflection, absorbtion을 다 고려해서 나온 것

## Lambertian Surface

이를 간단히 하기 위해서 입사각에 의해서 보든 것이 다 결정된다고 생각하는 surface가 바로 Lambertian surface!

### Lambert's law

![제목](/assets/images/Computer Vision/3-4.PNG){: width="70%" height="55%"}{: .align-center}

## others

이 외에도 여러 surface에 대한 모형이 있다

- Specular surface: 빛이 반사되면서 일종의 spread가 생기는 것(reflection angle에서 spread인듯)
- Phong model : 관측자가 바라보는 각도와 반사된 빛 간의 관계를 이용한 모형

## Formulation

![제목](/assets/images/Computer Vision/3-5.PNG){: width="70%" height="55%"}{: .align-center}

여기서의 문제는 reflected light과 source에 대한 정보는 알려져 있지만, surface에 대한 정보는 모르는 상태다!

![제목](/assets/images/Computer Vision/3-6.PNG){: width="70%" height="55%"}{: .align-center}

즉, 여기서 우리가 surface orientation에서 length를 고정하고 방향만 고려해서 2D로 생각할 수 있다.