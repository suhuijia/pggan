# PGGAN

### [[arxiv]](https://arxiv.org/abs/1710.10196) [[Official TF Project]](https://github.com/tkarras/progressive_growing_of_gans)

Authors : [Jihyeong Yoo](https://github.com/YooJiHyeong), [Daewoong Ahn](https://github.com/zsef123)


## How to use:

```
python3 main.py -h

usage: main.py [-h] [--gpus GPUS] [--cpus CPUS] [--save_dir SAVE_DIR]
               [--img_num IMG_NUM] [--optim_G {adam,sgd}]
               [--optim_D {adam,sgd}] [--loss {wgangp,lsgan}]
               [--start_resl START_RESL] [--end_resl END_RESL]
               [--beta [BETA [BETA ...]]] [--momentum MOMENTUM]
               [--decay DECAY] [--gp_lambda GP_LAMBDA]

PGGAN

optional arguments:
  -h, --help            show this help message and exit
  --gpus GPUS           Select GPU Numbering | 0,1,2,3 |
  --cpus CPUS           The number of CPU workers
  --save_dir SAVE_DIR   Directory which models will be saved in
  --img_num IMG_NUM     The number of images to be used for each phase
  --optim_G {adam,sgd}
  --optim_D {adam,sgd}
  --loss {wgangp,lsgan}
  --start_resl START_RESL
  --end_resl END_RESL
  --beta [BETA [BETA ...]]
                        Beta for Adam optimizer
  --momentum MOMENTUM   Momentum for SGD optimizer
  --decay DECAY         Weight decay for optimizers
  --gp_lambda GP_LAMBDA
                        Lambda as a weight of Gradient Panelty in WGAN-GP loss
```


#######################################
1. 训练数据预处理
cd ./datas
python3 preDataset.py    # 这里要记得修改代码最下面的原始图像路径
主要目的：将图像处理为 大小64，128， 256， 512，1024的图像，但是处理每个目标分辨率大小的原始图像必须大于该分辨率，也就是说只能缩小，不能做放大
举例：如果现在一张图片的最大边长为550，那么这张图片只能处理为64x64, 128x128, 256x256, 512x512, 不能够处理为1024x1024,因为对原始数据直接
     做插值放大，然后用来训练效果不佳，因为用来无监督的目标图像希望是高分辨高质量的。


2. 训练模型
python3 main.py   ## 默认参数

主要参数：gpus，使用gpu的编号，如果只有一个则只设置 “0”
        cpus，使用cpu的数量，根据自己机器设置
        img_num, 用于训练模型的图像数量
        start_resl, 最小分辨率设置，一般默认 4x4 不需要修改
        end_resl, 生成的最大分辨率，原作者为1024x1024, 由于你的数据量不足，而且原始图像也没有这么大分辨率无法设置，所以设置为512, 后续可根据情况设置256等

注意：main.py 中的 data_path 路径是读取数据集的路径，看清目录结构


3. 查看训练过程
pip install tensorboardX
tensorboard --logdir=./outs    ## ./outs 是训练输出的日志
然后根据控制台输出的地址输入浏览器查看，一般地址为：http://localhost:6006

