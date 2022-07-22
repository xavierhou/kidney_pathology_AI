# unet
## model
    unet_original按照论文的示意图实现
    1. dropout: "Drop-out layers at the end of the contracting path perform further implicit data augmentation."
    2. pad: valid模式导致边缘损失，concatenate之前先crop特征图
    3. up-conv: upsampling(rise resolution) + conv(reduce channels)
    4. last layer: "At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes"
    5. 这种策略主要是针对输入图片patch，带一圈上下文信息，优化分割结果
    
    unet_padding做了一点改动：
    1. 该模型针对整张图输入，使用same padding，保留边缘信息，输入输出尺寸相同，是对整幅图的预测
    2. 每一个卷积层后面接一个batch normalization，替换掉原论文的dropout
    3. 卷积block可以配置成residual

## training data
    1. membrane: 细胞1细胞壁0，512*512，30train+20test，注意正负样本不均匀
    2. disc: 间盘，512*512，71train+100test

## todolist
    1. 目前版本是single-class、single-label的，扩展多标签、多类别版本（激活函数、custom loss
    2. 衍生model：unet++ etc.

## multi-class & multi-label
    multi-class的mask是multi-channel的
    那么对应地，最后1*1 conv的channel维度应该是n_classes(single-class  --->  1)
    有两种情况：
    single-label：each pixel的各类别之间应该是竞争关系，那么activation可以用softmax，计算loss时可以加入交叉熵
    multi-label：有时物体之间有重叠，部分pixel拥有multi-label，那么activation应该用sigmoid，loss考虑二元交叉熵

## reweighting:
    reweighting bce: 正负样本unbalance
    bce_dice: 两个loss调整到相当的数量级
    reweighting dice: 可以对每个通道分别计算dice，然后加权求和

## inference:
    按照原论文的模型来实现，输入输出大小不同————valid padding的时候有边缘信息损失
    在做prediction时，为了预测整幅图的分割结果，要输入比原图更大尺寸的图，多出来的部分通过镜像来补全（cv2.copyMakeBorder）。

## 论文笔记:
    https://amberzzzz.github.io/2019/12/05/unet-vnet/


# vnet
## model
    按照原论文中的结构和维度来实现，3D换成2D，不关注原图的第三维
    residual: element-wise sum，前者channel数肯定多于后者，add之前先做zeropadding
    unlinearity: PReLU，最后输出的部分有点没看懂，
    shortcut: concatenate了compression path的feature map，
    output Layer: 1*1 conv+PReLU，output channel，softmax

## 相比较于unet:
    residual
    diceloss

## focal loss nan:
    nan问题————当对过小的数值进行log操作，返回值将变为nan
    解决：clamp

## todolist:
    settings: multi-class & multi-channel & multi-label
    data: preparation and augmentation
    1.10 added: gaussian mask for line objects and shift&flip augmentation

## add & concatenate:
    add操作是by element相加，要求两个输入shape完全相同，如果不同，先zero padding
    concatenate操作是在channel维度的stack，要求两个输入其他维度的shape相同，channel维度可以不同

## experiment
    就我这边颈椎X光片下颌线、上颚骨以及椎块的分割来看，同样的实验设置下，**本工程**的vnet模型test结果要略好于unet。
    vnet学习位置信息更强，mask更完整（unet比较碎）。

## batch normalization & activation
    注意顺序：linear layer - BN layer - unlinear layer
    e.g. conv / dense - BN - relu

## 激活函数:
    PReLU会显著增加参数量

## kernel size:
    3x3替换5x5参数量会显著减少
    orig_unet(3x3), orig_vnet(5x5)

## loss
    实验目标分割背景、两条线和一个块状目标，
    发现对于块状目标，focal_dice的边缘分割效果好于bce_dice，
    对于线状目标，bce_dice在localization的方面好于focal_dice(后者会在别的地方出现假阳)，focal_dice的检出率好于bce_dice，最终决定用focal_bce_dice loss。
    另外focal loss和dice loss的线性叠加(之前用的是focal+log(dice))以后，loss会出现震荡，收敛效果不好，但是能观察到dice逐渐提升。


# 衍生网络
## backboned-unet:
    1. better feature extracting blocks
    2. using pre-trained backbone & weights: 不同的keras版本下，resnet backbone不一样，summary发现差别在最后有没有接一个avg_pool
    3. 不要随随便便搬过来一个网络结构替换原有的backbone，如resnet50，因为resnet是为了分类任务设计，大感受野这种性质在分割任务中没好处
    4. 但是可以尝试替换局部的block（引入多尺度&减少参数）
    5. newly added: Unets with Resnet 34 encoders, kaggle TGS Salt Challenge proved有效

## fine-grained-unet:
    1. 考虑到一些细粒度的instance，在roi内分割效果更好，而我们恰好能够提供ROI。
        尝试双输入&双输出，一边输入full image，一边输入ROI image，一边输出big instance，一边输出tiny instance。
        在训练阶段，由于有先验知识(ROI)，我们可以并行训练，
        在测试阶段，我们进行串行预测，首先输入全图，获得big instance的mask，然后基于big mask获得ROI输入，获得tiny instance mask。
        为了共享网络，crop出的ROI要resize成全图尺寸
    2. crop的两种方式
        crop square
        crop mask
    3. loss: 新任务loss测试下来dice<dice+bce<dice+reweighting_bce<dice+reweighting_bce+focal
        总之就是加权的bce太牛逼了

## branch2-unet:
    multi-channel的输出，custom的loss，很可能使得网络在特征筛选的阶段对某个／几个任务有倾向性。
    我们无法量化这种影响，但是可以通过特征图heatmap看到这种倾向，
    一个解决的方案是，比如观察到某个通道的任务，在level2上特征响应很强，在后面几层没了，
    那么我们就直接在这一层，对这个任务做特征筛选，然后上采样，叠加到原来的输出上。

## 3d-unet:
    3D版本的unet，conv_block的第二个conv做了double channel

## 3d-vnet:
    真正的vnet论文实现
    depth=5, residual use element-sum, shortcut use concate
    5x5 conv kernel

## 3d-vnet-depthwise:
    用可分离卷积替换标准卷积，用stride2 conv替换Max pooling
    参数量：depth=5 kernel_size=3
    3d-vnet: 14,324,241      3d-vnet-depthwise: 487,601     差了一个数量级
    zeropadding in conv_block: keras的mobineNet源码里面，在实现depthwise block的时候，先做了zeropadding，再做了valid conv
    直接same pad也work，主要影响在模型转换(https://github.com/starhiking/Document/blob/master/Deep_Learning/Note/Pad_Difference.md)

## vnet3d:
    目前实验下来效果最好的vnet结构

## unet2.5d
    复现论文：Automatic Segmentation of Vestibular Schwannoma from T2-Weighted MRI by Deep Spatial Attention with Hardness-Weighted Loss
    论文写的贼水，不知道是不是真的有用
    另外prelu参数量太大，得换成别的激活函数






