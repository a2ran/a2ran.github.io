---
title: "[Paper] Self-Supervised Visual Feature Learning With Deep Neural Networks: A Survey"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
  - Self Supervised
tags:
  - [survey]
---

"Self-Supervised Visual Feature Learning With Deep Neural Networks: A Survey"이란 논문에 대한 리뷰입니다.

원문은 [링크](https://ieeexplore.ieee.org/abstract/document/9086055?casa_token=JYQCWVwcqGMAAAAA:sk6ej5aC1PYzRk85Tgd3lUtmSQUG88LdZRKZYQBM3vVs3nfyGRP_QV_dOhfjzuOisUbt2H8)에서 확인할 수 있습니다.

# Insight
-	First train general attributes
 - Shallower blocks focus on the low-level general features
 - Deeper blocks focus on the high-level task specific features

-	Effective pretext tasks ensure that semantic features learned through the process of accomplishing the pretext tasks.

Pseudo labels without human annotations
-	Minimizing error between the prediction & pseudo labels

# Schema

## Pre-training with pretext 
- (Unlabelled, Self-supervised)	Generation based
 - Context based
 - Free semantic label based
 - Cross modal based

## Downstream Task (Labelled)	Image Classification
- Sematic segmentation
- Object detection
- Human action recognition

# Feature Learnings description
- Generation based	Image Generation	GAN
 - Video Generation	
- Context based	Context Similarity	Between image patches
	- Spatial Context Similarity	Spatial relation among image patches
	- Temporal Context Similarity	Frame sequence (Video)
- Free semantic label based	
- Cross modal based	Two different channels of input data correspond to each other

# Downstream Task Describe
- Image Classification (Quality of feature)	
- Sematic segmentation (Generality of feature)	
 -Assign semantic labels to each pixel (Pixelwise annotation)
- Object detection (Generality of feature)
 - Localizing the position of objects
  -	Proposals based on feature map
  -	FC to bound box of object
- Human action recognition (Quality of feature)	

# Feature Learning
- Generation based	
  - Pseudo labels are images themselves
- Context based	
  - (Common) clustering
  - Predictive task for group ID
  - Contrastive task for distance btw features
- Free semantic label based	
- Cross modal based	

# Qualitative evaluation
- Kernel Visualization	Compare first convolution layer with supervised one
- Feature Map Visualization	Attention of networks


