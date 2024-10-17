<div align="center">

## 👉 English | [简体中文](readme_cn.md)

</div>

### 1 Introduction

Implementing face recognition algorithms on CIFAR-100 and CIFAR-10 dataset

- Motivation

  - I find out that popular face datasets in papers have many identities and massive images.
    Furthermore, some pre-processing procedures are required before we utilize them to train our models or implement our algorithms.
- My Targets

  - I want to have a simple impression about their performance. Then I decided to implement them on CIFAR-100 and CIFAR-10 dataset.

### 2 Contents

* Softmax Loss  (Deep face recognition, BMVC2015)

setup as a N-ways classification problem

* Triplet Loss   (FaceNet CVPR2015)

Facenet: A unified embedding for face recognition and clustering

* ArcFace (CVPR 2019)

Arcface: Additive angular margin loss for deep face recognition

Note: Adding margin to the angle of cosin similarity between features and the weights of the last fully connected layer.

* ElasticFace (CVPRW 2022)

Elasticface: Elastic margin loss for deep face recognition

Note: Changing the fixed margin value in ArcFace as sampling from gaussian distribution. The larger angles are associated with smaller margin values, and vice versa.

* UniFace (ICCV 2023)

Uniface: Unified cross-entropy loss for deep face recognition

### 3 Experimental Results

* CIFAR100

| Loss Type    | Backbone | Accuracy (%) | EER (%) |
| ------------ | -------- | ------------ | ------- |
| Softmax      | Resnet18 | 76.31        | 4.78    |
| Triplet Loss | Resnet18 | 71.43        | 6.58    |
| ArcFace      | Resnet18 | 76.03        | 7.91    |
| ElasticFace  | Resnet18 | 76.41        | 9.9     |
| UniFace      | Resnet18 | 74.53        | 10.94   |

* CIFAR10

| Loss Type    | Backbone | Accuracy (%) | EER (%) |
| ------------ | -------- | ------------ | ------- |
| Softmax      | Resnet18 | 93.71        | 3.25    |
| Triplet Loss | Resnet18 | 94.01        | 2.96    |
| ArcFace      | Resnet18 | 93.77        | 2.92    |
| ElasticFace  | Resnet18 | 94.34        | 3.16    |
| UniFace      | Resnet18 | 91.72        | 8.85    |

### 4 References

- Parkhi, O., Vedaldi, A., & Zisserman, A. (2015). Deep face recognition. In BMVC 2015-Proceedings of the British Machine Vision Conference 2015. British Machine Vision Association.
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 815-823).
- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4690-4699).
- Boutros, F., Damer, N., Kirchbuchner, F., & Kuijper, A. (2022). Elasticface: Elastic margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 1578-1587).
- Zhou, J., Jia, X., Li, Q., Shen, L., & Duan, J. (2023). Uniface: Unified cross-entropy loss for deep face recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 20730-20739).

### 5 Cite this repository

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

```RSMung
@Misc{simple_face_recognition,
howpublished = {\url{https://github.com/RSMung/SimpleFaceRecognition}},   
title = {Implementing face recognition algorithms on CIFAR-100 and CIFAR-10 dataset},  
author = {RSMung}  
}  
```
