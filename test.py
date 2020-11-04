import os, sys
import argparse
print(sys.version)
import torch
import torch.nn as nn
from models.Generator import G
import numpy as np
import matplotlib.pyplot as plt
import cv2


def arg_parse():
    parser = argparse.ArgumentParser(description="PGGAN")

    parser.add_argument('--use_gpu', type=bool, default=False,
                        help="whether use cuda")

    parser.add_argument('--model_file', type=str, default="outs/step_0000044_resl_8_stabilization.pth.tar",
                        help="inference need to weight file")

    parser.add_argument('--save_dir', type=str, default='./genImage',
                        help='Directory which generate images will be saved in')

    parser.add_argument('--input_normalize', action="store_true", help="normalize input range to [0, 1]")
    parser.add_argument('--start_resl', type=float, default=4)
    parser.add_argument('--end_resl', type=float, default=512)
    
    return parser.parse_args()


def normalization(x):
    """"
    归一化到区间{0,1]
    """
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range


class GenImage(object):
    """docstring for GenImage"""
    def __init__(self, args):
        super(GenImage, self).__init__()
        self.model_file = args.model_file
        self.use_gpu = args.use_gpu
        self.start_resl = args.start_resl
        self.save_dir = args.save_dir
        self.start_resl = args.start_resl
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def load_model(self):
        """Load generated model file"""
        if self.model_file is None:
            print("argument 'model_file' is error.")
            return None

        if self.use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        if os.path.exists(self.model_file):
            self.Gmodel = nn.DataParallel(G()).to(self.device)
            print(self.Gmodel)
            print("Lode model file: %s" % self.model_file)
            ckpoint = torch.load(self.model_file, map_location=self.device)
            self.load_resl = ckpoint["resl"]
            resl = self.start_resl
            while resl < self.load_resl:
                self.Gmodel.module.grow_network()
                self.Gmodel.to(self.device)
                resl *= 2

            self.Gmodel.load_state_dict(ckpoint["G"])
            print("Load model done...")

        else:
            print("Load model fail...")
            return None

        return True


    def genImage(self, img_num=1, mode="stabilization"):
        """generate images from noise
        args: img_num: int, the number of generate images
               mode: str, model type, include 'stablization', 'transition', must consistent with model
        """
        self.Gmodel.eval()
        with torch.no_grad():
            for i in range(img_num):
                latent_z = torch.randn(1, 512, 1, 1).normal_().to(self.device)
                output = self.Gmodel(latent_z, mode)
                print("output size: ", output.size())
                output = torch.clamp(output, min=0, max=1)
                output = output.cpu().squeeze().numpy()
                fake_img = output.transpose(1, 2, 0)
                print("fake image size: ", fake_img.shape)
                plt.imshow(fake_img)
                plt.show()
                save_file = os.path.join(self.save_dir, str(self.load_resl), "%05d.jpg" % i)
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                plt.imsave(save_file, fake_img)

        

if __name__ == "__main__":

    ## 参数配置
    args = arg_parse()

    ## 模型初始化
    GI = GenImage(args=args)

    ## 模型加载
    model = GI.load_model()

    ## 生成图片
    if model is None:
        print("don't load model...")
    else:
        print("model init done...")
        GI.genImage(img_num=2)




    
