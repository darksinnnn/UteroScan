import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize


from pathlib import Path
from .preprocessing import preprocess_ultrasound
from .augmentation import get_ultrasound_train_transforms


class DataGenerator(data.Dataset):
    def __init__(self, annotation_lines, input_shape, random=True, autoaugment_flag=True):
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.random             = random
        
        # Phase 2: Clinical ultrasound augmentation pipeline for training only
        self.train_transforms   = get_ultrasound_train_transforms() if self.random else None

    def __len__(self):
        return len(self.annotation_lines)

    def _resolve_path(self, path_str):
        p = Path(path_str)
        if p.is_file():
            return str(p)
        candidates = [
            Path("datasets") / p,
            Path("..") / ".." / "datasets" / "datasets" / Path(*p.parts[1:] if len(p.parts) > 1 else p.parts),
            Path("d:/Project/UteroScan/datasets/datasets") / Path(*p.parts[1:] if len(p.parts) > 1 else p.parts),
            Path("d:/Project/UteroScan/datasets/datasets") / p,
            Path("d:/Project/UteroScan") / p
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        return path_str

    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        resolved_path = self._resolve_path(annotation_path)
        image = Image.open(resolved_path)
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image = cvtColor(image)

        #--------------------------------------------------------------#
        #   Phase 1: 临床超声预处理 (224x224 INTER_AREA, 中值滤波降斑, CLAHE对比度增强)
        #--------------------------------------------------------------#
        image = preprocess_ultrasound(image, target_size=(self.input_shape[1], self.input_shape[0]))

        #--------------------------------------------------------------#
        #   Phase 2: 临床超声数据增强 (仅应用于训练集: 旋转±15°, 水平翻转, 缩放0.9-1.1, 错切±10°; 严禁垂直翻转)
        #--------------------------------------------------------------#
        if self.random and self.train_transforms is not None:
            image = Image.fromarray(image)
            image = self.train_transforms(image)
            image = np.array(image)

        image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])

        y = int(self.annotation_lines[index].split(';')[0])
        return image, y

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
    
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        #------------------------------------------#
        #   resize并且随即裁剪
        #------------------------------------------#
        image = self.resize_crop(image)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   随机增强
        #------------------------------------------#
        image = self.policy(image)
        return image
            
def detection_collate(batch):
    images = []
    targets = []
    for image, y in batch:
        images.append(image)
        targets.append(y)
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    targets = torch.from_numpy(np.array(targets)).type(torch.FloatTensor).long()
    return images, targets
