**# aug-efficientdet**

> Architecture is Forked from [zylo117](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) 

## What will be in this repository?

- [ ] Dataset Analysis - BDD100K aiming at Object Dectection in self-driving 

- [ ] Data Augmentation of EfficientDet Pytorch Version

    - [x] Mosaic (special methods based on specific datasets)

        `efficientdet/dataset.py`

- [x] Rewrite the [original](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) data structure (dataloader - using DarkNet data form)

    `efficientdet/utils.py` `efficientdet/loss.py` `efficientdet/dataset.py`

    `efficientdet_test.py`

- [ ] Overhaul Efficientdet codes in Pytorch 

## Mosaic Methods

![Mosaic for BDD100K.jpg](https://site-pictures.oss-eu-west-1.aliyuncs.com/y2wpc.jpg)

## Results on TensorFlow official version:

![image-20200518225201539](https://site-pictures.oss-eu-west-1.aliyuncs.com/0t897.png)

![image-20200518225251433](https://site-pictures.oss-eu-west-1.aliyuncs.com/93f7h.jpg)

![image-20200518225319504](https://site-pictures.oss-eu-west-1.aliyuncs.com/dvbnp.jpg)

![image-20200518225452006](https://site-pictures.oss-eu-west-1.aliyuncs.com/9j5s5.jpg)

