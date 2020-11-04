import os, glob
import random
from PIL import Image
import traceback


class PreDataset:
    def __init__(self, orig_path):
        self.orig_path = orig_path
        ## nitoce！if you need image size 512 or 1024，the origin image must be more than 512 or 1024
        self.imgSize_list = [64, 128, 256, 512, 1024]
        self.dataset_name = "sewer"
        for size in self.imgSize_list:
            size_str = str(size)
            save_path = os.path.join(os.path.dirname(orig_path), "dataset-%s"%size_str, self.dataset_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

    def predata(self):
        ## read origin image
        if not os.path.exists(self.orig_path):
            raise "image directory %s not exist..." % self.orig_path

        try:
            origImgList = os.listdir(self.orig_path)
            for img_file in origImgList:
                print(img_file)
                img_path = os.path.join(self.orig_path, img_file)
                pil_img = Image.open(img_path).convert("RGB")
                print(pil_img.mode)
                imgmax = max(pil_img.size)
                print("image the longest edge: ", imgmax)
                for imgSize in self.imgSize_list:
                    print("scale size: ", imgSize)
                    if imgmax >= imgSize:
                        temp_img = pil_img.resize((imgSize, imgSize))
                        save_path = os.path.join(os.path.dirname(orig_path), "dataset-%s"%imgSize, self.dataset_name, img_file)
                        temp_img.save(save_path)

        except Exception as e:
            traceback.format_exc(e)



if __name__ == '__main__':

    ### dataset directory path
    orig_path = "../dataset/dataset"
    print(os.path.dirname(orig_path))

    PD = PreDataset(orig_path)
    PD.predata()

