<div align="center">

## [简体中文](readme_cn.md) | English

</div>

#### 1 Implementing face recognition algorithms on CIFAR-100 dataset

I find out that popular face datasets in papers have many identities and massive images.

Furthermore, some pre-processing procedures are required before we utilize them to train our models or implement our algorithms.

I just want to have a simple impression about their performance. Then I decided to implement them on CIFAR-100 dataset.

#### 2 Contents

* Softmax Loss
* Triplet Loss
* ArcFace (CVPR 2019)
* ElasticFace (CVPRW 2022)
* UniFace (ICCV 2023)

#### 3 Experimental Results

* CIFAR100

| Loss Type   | Backbone | Accuracy (%) | EER (%) |
| ----------- | -------- | ------------ | ------- |
| Softmax     | Resnet18 |              |         |
| ArcFace     | Resnet18 |              |         |
| ElasticFace | Resnet18 |              |         |
| UniFace     | Resnet18 |              |         |

#### 4 References

ArcFace: Additive Angular Margin Loss for Deep Face Recognition
