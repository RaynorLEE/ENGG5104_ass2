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
        padded_img = PIL.Image.new(mode=img.mode, size=tuple([img.size[0] + self.padding * 2,
                                                              img.size[1] + self.padding * 2]), color=0)
        padded_img.paste(img, tuple([self.padding, self.padding]))
        return padded_img


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, **kwargs):
        H, W = img.size
        y = random.randint(0, H - self.size)
        x = random.randint(0, W - self.size)
        cropped_img = img.crop(box=tuple([x, y, x + self.size, y + self.size]))
        return cropped_img


class RandomFlip(object):
    def __init__(self, ):
        pass

    def __call__(self, img, **kwargs):
        # Use Horizontal flipping only is better than use horizontal and vertical flipping
        h = random.randint(0, 1)
        v = random.randint(0, 1)
        if h == 1:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        # if v == 1:
        #     img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        return img