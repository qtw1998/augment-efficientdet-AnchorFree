# aug-efficientdet
title: EfficientNet🍭EfficientDet 
date: 2019-12-11T01:06:17.191Z
categories: [DeepLearning]
img: https://site-pictures.oss-eu-west-1.aliyuncs.com/2p69f.jpg
mathjax: true
summary: ModelCompression
tags: 

    - Theory
        - CNN
    
    - ModelComression

---

Reading List:

**EfficientNet** **FPN** **EfficientDet** **DetNAS**

## EfficientNet

> **balance all dimensions** of network width/depth/resolution
>
> uniformly scaling each of them with **constant ratio**
>
> **<u>Result</u>**: EfficientNet models generally use an order of **magnitude fewer parameters and FLOPS** than other ConvNets with similar accuracy.
>
> *compound scaling method* 

### Necessity

<img src="https://site-pictures.oss-eu-west-1.aliyuncs.com/d5oba.jpg" alt="Scaling Methods" style="zoom:50%;" />

Intuitively, the compound scaling method makes sense because if <u>the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image</u>.

直观地说，复合缩放方法是有意义的，因为**<u>如果输入图像更大，那么网络需要更多的层来增加感受野，需要更多的通道来捕捉更大图像上更细粒度</u>**的模式。事实上，之前的理论和实证结果都表明网络宽度和深度之间存在一定的关系，但据作者所知，作者是第一个对网络宽度、深度和分辨率三个维度之间的关系**进行实证量化**的人

### Compound Model Scaling

#### Problem Formulation

##### ConvNet

<img src="https://site-pictures.oss-eu-west-1.aliyuncs.com/qg9vs.jpg" alt="ConvNet Function" style="zoom:45%;" />

##### Optimization

<img src="https://site-pictures.oss-eu-west-1.aliyuncs.com/as5i3.jpg" alt="Constricts" style="zoom:45%;" />

##### Scaling Dimensions

- Depper ConvNet : capture richer & more complex features & generalize well on new tasks
- Wider networks: capture more fine-grained features & easier to train
- Resolution: capture .. & improve accuracy 

<img src="https://site-pictures.oss-eu-west-1.aliyuncs.com/cg1bb.png" alt="Parameters" style="zoom:45%;" />

![CAM-EfficientNet](https://site-pictures.oss-eu-west-1.aliyuncs.com/0c494.jpg)

## FPN - Feature Pyramid Network

> - 核心亮点：**提出了多尺度的特征金字塔结构**
>
>     将**最后一层**特征图谱进行**不断进行上采样**, 并与每一个金字塔阶级的特征图谱进行加法合并操作, 得到新的**表征能力更强的不同金字塔层次的特征图谱**, 然后按照尺寸分别映射到这些特征图谱上, 再在每个特征图谱上进行类别和位置预测
>
> - 基本思想: 利用不同level的feature map预测不同尺度的目标
>
>     - 尺寸小的物体因不断的池化会在较深的层消失，所以利用**<u>浅层检测小目标</u>**
>     - 浅层不如**<u>深层具备丰富的语义特征</u>**，所以还需要浅层融合深层的特征
>     - FPN = top-down的融合（skip layer） + 在金字塔各层进行prediction

<img src="https://site-pictures.oss-eu-west-1.aliyuncs.com/wbc0b.jpg" alt="Featureized image pyramid" style="zoom:35%;" />



multi-scale feature fusion 

SSD较早尝试了使用CNN金字塔形的层级特征。理想情况下，SSD风格的金字塔 重利用了前向过程计算出的来自多层的多尺度特征图，因此这种形式是不消耗额外的资源的。但是SSD为了避免使用low-level的特征，放弃了浅层的feature map，而是从conv4_3开始建立金字塔，而且加入了一些新的层。因此SSD放弃了重利用更高分辨率的feature map，但是这些feature map对检测小目标非常重要。**这就是SSD与FPN的区别**。



FPN是为了自然地利用CNN层级特征的金字塔形式，**同时生成在所有尺度上都具有强语义信息的特征金字塔**。所以FPN的结构设计了top-down结构和横向连接，以此融合具有高分辨率的浅层layer和具有丰富语义信息的深层layer。这样就实现了从单尺度的单张输入图像，快速构建在所有尺度上都具有强语义信息的特征金字塔，同时不产生明显的代价。

## EfficientDet

> EfficientDet-D7 achieves state-of-the-art accuracy with an order-of-magnitude **fewer parameters and FLOPS** than the best existing detector. 
>
> EfficientDet is also up to <u>3.2x faster on GPUs</u> and <u>8.1x faster on CPUs</u>.
>
> † BiFPN & † Compound Scaling
>
> 可学习的**<u>权重</u>**来学习不同输入特征的重要性，同时重复应用自上而下和自下而上的<u>**多尺度特征融合**</u>

### ^†^ BiFPN

BiFPN(weighted bi-directional feature pyramid network)  加权双向特征金字塔网络

Our final BiFPN integrates both the **bidirectional cross- scale connections** and the **fast normalized fusion**. 

BiFPN 引入两个主要想法：**高效的双向跨尺度连接**和**加权特征融合**

bidirectional (top-down & bottom-up)  & cross-scale connections & weighted feature fusion.

#### Problem Formulation

![FPN Unit](https://site-pictures.oss-eu-west-1.aliyuncs.com/vh9ok.jpg)

注：其中 Resize 通常表示**分辨率匹配**时的上采样或者下采样，Conv 通常表示特征处理时的卷积网络

#### Cross-Scale Connections

<img src="https://site-pictures.oss-eu-west-1.aliyuncs.com/s8bth.jpg" alt="Evolution to BiFPN" style="zoom:40%;" />

b）*PANet 在 FPN 的基础上额外添加了自下而上的路径*

*c）NAS-FPN 使用神经架构搜索找出不规则**特征网络拓扑**；*

*d)  在所有输入特征和输出特征之间添加成本高昂的连接；*

*e ) 移除只有一个输入边的节点，从而简化 PANet；*

if a node has only one input edge with no feature fusion, then it will have less contribution to feature network that aims at fusing different features.移除仅具备一个输入边的节点。其背后的想法很简单：如果一个节点只有一个输入边**没有特征融合**，则它**对特征网络的贡献较小**。这样就得到了简化版 PANet

**<u>*f ) 是兼顾准确和效率的 BiFPN*</u>** 

add an extra edge from the original input to output node if they are at the same level, in order to fuse more features without adding much cost为**同级原始输入**到输出节点添加额外的边，从而在不增加大量成本的情况下融合更多特征

treat each <u>bidirectional (top-down & bottom-up)</u> path as **one** <u>feature network layer</u>, and **repeat** the same layer multiple times to <u>enable more high-level feature fusion</u>.与只有一条自上而下和自下而上路径的 PANet 不同，将每个双向路径（自上而下和自下而上）作为一个特征网络层，并多次**重复同一个层**，以实现更高级的特征融合

#### Weighted Feature Fusion 加权融合特征

Previous feature fusion methods treat all input features equally without distinction. 

由于不同输入特征的**分辨率不同**，它们**对输出特征的贡献**也不相等

different input features are at different resolutions, they usually contribute to the output feature 

add an additional weight for each input during feature fusion, and let the network to <u>learn the importance of each input feature *unequally*</u>

<img src="https://site-pictures.oss-eu-west-1.aliyuncs.com/13v5b.jpg" alt="Fast normalized fusion Function" style="zoom:40%;" />

similar learning behavior and accuracy as the <u>softmax-based fusion</u>

runs up to 30% faster on GPUs

$P_{6}^{t d}$ : intermediate feature at level 6 , $P_{6}^{out}$ is the output feature at level 6 on the bottom-up pathway

<img src="https://site-pictures.oss-eu-west-1.aliyuncs.com/9d1vq.jpg" alt="BiFPN Unit" style="zoom:50%;" />

Notably, to further improve the efficiency, we use **depthwise separable convolution** for feature fusion, and **add batch normalization** and **activation after each convolution**.

### EfficientDet Network

#### EfficientDet Architecture

<img src="https://site-pictures.oss-eu-west-1.aliyuncs.com/2p69f.jpg" alt="EfficientDet Architecture" style="zoom:50%;" />

​           Backbone network -                     -feature network -                           shared class/box prediction network 

**基于 BiFPN**，开发了新型检测模型 EfficientDet。上图展示了 EfficientDet 的**整体架构**，遵循one-stage范式

使用 <u>EfficientNet</u> 作为**主干网络**，使用 <u>BiFPN</u> 作为**特征网络**，并**使用共享的边界框/类别预测网络**

在 ImageNet 数据集上预训练的 EfficientNet 作为**主干网络**，将 BiFPN 作为**特征网络**，接受来自主干网络的 level 3-7 特征 {P3, P4, P5, P6, P7}，并重复应用自上而下和自下而上的**双向特征融合**。然后将融合后的特征输入**边界框/类别预测网络**，分别输出目标类别和边界框预测结果

#### ^†^ Compound Scaling

A key challenge here is how to scale up a baseline EfficientDet model.

> **scaling up all dimensions**    
>
> jointly scale up all dimensions of backbone network, BiFPN network, class/box network, and resolution
>
> 同时对所有主干网络 & 特征网络 & 边界框/类别预测网络 & 分辨率  执行统一缩放

使用简单的复合系数 φ 统一扩大主干网络、BiFPN 网络、边界框/类别预测网络的所有维度

## DetNas

### **NAS: Neural Architecture Search**

**自动化神经架构搜索** 

**One-Shot** 只完整训练一个超网，主张权重共享，对不同路径进行采样作为子模型训练，并基于此对模型排名，这样就可以更快速判断模型性能，提高搜索效率

> 和DetNet目的相似,寻找更合适的目标检测backbone网络