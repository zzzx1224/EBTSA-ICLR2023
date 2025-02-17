import numpy as np
from PIL import Image
import random
import torch
from PIL import ImageEnhance
from PIL import ImageFilter

class random_rot(object):
    def __init__(self, d):
        self.d = d
    def __call__(self, img):
        p = np.random.random()
        if p < 1.0:
            f = np.random.randint(-self.d,self.d)
            img = img.rotate(-f)
        return img

class random_resize(object):
    def __init__(self, min_value, max_value, min_size):
        self.min_value = min_value
        self.max_value = max_value
        self.min_size = min_size
    def __call__(self, img):
        min_size = self.min_size
        scale = np.random.uniform(self.min_value, self.max_value)
        size = img.size[0]

        size_new = int(scale * size)

        img = img.resize((size_new, size_new))

        if size_new < min_size:
            canvas = Image.new('RGB', (min_size,min_size), 0)
            x = random.randint(0, min_size - size_new)
            y = random.randint(0, min_size - size_new)
            canvas.paste(img, (x,y))
            img = canvas

        return img

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

jitter_param  = dict(Brightness=0.4, Contrast=0.4, Color=0.4)
transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out

#添加椒盐噪声
class AddSaltPepperNoise(object):

    def __init__(self, density=0,p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  # 概率的判断
            img = np.array(img)  # 图片转numpy
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
            mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
            img[mask == 0] = 0  # 椒
            img[mask == 1] = 255  # 盐
            img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
            return img
        else:
            return img

#添加Gaussian噪声
class AddGaussianNoise(object):

    '''
    mean:均值
    variance：方差
    amplitude：幅值
    '''
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):

        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

#添加模糊
class Addblur(object):

    def __init__(self, p=0.5,blur="normal"):
        #         self.density = density
        self.p = p
        self.blur= blur

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  # 概率的判断
             #标准模糊
            if self.blur== "normal":
                img = img.filter(ImageFilter.BLUR)
                return img
            #高斯模糊
            if self.blur== "Gaussian":
                img = img.filter(ImageFilter.GaussianBlur(radius=5))
                return img
            # contour
            if self.blur== "CONTOUR":
                img = img.filter(ImageFilter.CONTOUR)
                return img
            #均值模糊
            if self.blur== "mean":
                img = img.filter(ImageFilter.BoxBlur)
                return img

        else:
            return img

class Addfilter(object):

    def __init__(self, p=0.5,blur="normal"):
        #         self.density = density
        self.p = p
        self.blur= blur

    def __call__(self, img):
            p1 = random.uniform(0, 1)
            if p1 < 0.2:
                img = img.filter(ImageFilter.GaussianBlur(radius=5))
                return img
            elif p1 >= 0.2 and p1 < 0.4:
                img = img.filter(ImageFilter.CONTOUR)
                return img
            elif p1 >= 0.4 and p1 < 0.6:
                img = img.filter(ImageFilter.FIND_EDGES)
                return img.convert('RGB')
            elif p1 >= 0.6 and p1 < 0.8:
                return img.convert('1').convert('RGB')
            else:
                return img.convert('L').convert('RGB')