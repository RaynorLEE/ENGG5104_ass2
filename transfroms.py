import torch
import numpy as np
import PIL
import random
import cv2


# TODO: implementation transformations for task3;
# You cannot directly use them from pytorch, but you are free to use functions from cv2 and PIL
class Padding(object):
    def __init__(self, padding: int):
        self.padding = padding

    def __call__(self, img, **kwargs):
        # padded_img = np.zeros([img.size[0]+self.padding, img.size[1]+self.padding], 3, dtype=)
        padded_img = PIL.Image.new(mode=img.mode, size=tuple([img.size[0] + self.padding, img.size[1] + self.padding]),
                                   color=0)
        padded_img.paste(img, tuple([self.padding, self.padding]))
        return padded_img


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, **kwargs):
        y = random.randint(0, img.size[0] - self.size - 1)
        x = random.randint(0, img.size[1] - self.size - 1)
        cropped_img = img.crop(box=tuple([x, y, x+img.size[1], y+img.size[0]]))
        return cropped_img


class RandomFlip(object):
    def __init__(self, ):
        pass

    def __call__(self, img, **kwargs):
        h = random.randint(0, 1)
        v = random.randint(0, 1)
        if h == 1:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        # if v == 1:
        #     img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        return img
