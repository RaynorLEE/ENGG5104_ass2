import torch
import numpy as np
import PIL
import random
import cv2
from PIL import ImageOps

# TODO: implementation transformations for task3;
# You cannot directly use them from pytorch, but you are free to use functions from cv2 and PIL
# class Padding(object):
#     def __init__(self, padding: int):
#         self.padding = padding
#
#     def __call__(self, img, **kwargs):
#         # padded_img = np.zeros([img.size[0]+self.padding, img.size[1]+self.padding], 3, dtype=)
#         padded_img = PIL.Image.new(mode=img.mode, size=tuple([img.size[0] + self.padding * 2,
#                                                               img.size[1] + self.padding * 2]), color=0)
#         padded_img.paste(img, tuple([self.padding, self.padding]))
#         return padded_img
#
#
# class RandomCrop(object):
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, img, **kwargs):
#         y = random.randint(0, img.size[0] - self.size)
#         x = random.randint(0, img.size[1] - self.size)
#         cropped_img = img.crop(box=tuple([x, y, x+img.size[1], y+img.size[0]]))
#         return cropped_img
#
#
# class RandomFlip(object):
#     def __init__(self, ):
#         pass
#
#     def __call__(self, img, **kwargs):
#         h = random.randint(0, 1)
#         v = random.randint(0, 1)
#         if h == 1:
#             img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
#         # if v == 1:
#         #     img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
#         return img

class Padding(object):
    def __init__(self,padding):
        self.size = padding

    def __call__(self, img, **kwargs):
        img = ImageOps.expand(img, border=(self.size, self.size, self.size, self.size), fill=0)
        return img


class RandomCrop(object):
    def __init__(self,size):
        self.crop_size = size
    def __call__(self, img, **kwargs):
        W = img.size[0]
        H = img.size[1]
        Wl = random.randint(0,W - self.crop_size)
        Wr = W - self.crop_size - Wl
        Ht = random.randint(0,H - self.crop_size)
        Hd = H - self.crop_size - Ht
        img = ImageOps.crop(img, border=(Wl, Ht, Wr, Hd))
        return img


class RandomFlip(object):
    def __init__(self,):
        pass
    def __call__(self, img, **kwargs):
        seed = random.random()
        if seed<0.45:
            img = img
        elif 0.45<= seed and seed<=0.5:
            img = ImageOps.flip(img)
        elif seed>0.55:
            img = ImageOps.mirror(img)
        else:
            img = ImageOps.flip(img)
            img = ImageOps.mirror(img)

        return img
