import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import pdb


class GaussianBlur(object):

    def __init__(self, min=0.1, max=2.0, kernel_size=9):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class ReplayBuffer(object):
    # def __init__(self, size, transform, dataset):
    def __init__(self, size, num_classes):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = {}
        for i in range(num_classes):
            self._storage[i] = []
        self._maxsize = size
        self.num_classes = num_classes
        self._next_idx = 0
        self.gaussian_blur = GaussianBlur()

        # def get_color_distortion(s=1.0):
        # # s is the strength of color distortion.
        #     color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
        #     rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        #     rnd_gray = transforms.RandomGrayscale(p=0.2)
        #     color_distort = transforms.Compose([
        #         rnd_color_jitter,
        #         rnd_gray])
        #     return color_distort

        # color_transform = get_color_distortion()

        # if dataset == "cifar10":
        #     im_size = 32
        # elif dataset == "continual":
        #     im_size = 64
        # elif dataset == "celeba":
        #     im_size = 128
        # elif dataset == "object":
        #     im_size = 128
        # elif dataset == "mnist":
        #     im_size = 28
        # elif dataset == "moving_mnist":
        #     im_size = 28
        # elif dataset == "imagenet":
        #     im_size = 128
        # elif dataset == "lsun":
        #     im_size = 128
        # else:
        #     assert False

        # self.dataset = dataset
        # if transform:  # multi-level ebm
        #     if dataset == "cifar10":
        #         self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
        #     elif dataset == "continual":
        #         color_transform = get_color_distortion(0.1)
        #         self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.7, 1.0)), color_transform, transforms.ToTensor()])
        #     elif dataset == "celeba":
        #         self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
        #     elif dataset == "imagenet":
        #         self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.01, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
        #     elif dataset == "object":
        #         self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.01, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
        #     elif dataset == "lsun":
        #         self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
        #     elif dataset == "mnist":
        #         self.transform = None
        #     elif dataset == "moving_mnist":
        #         self.transform = None
        #     else:
        #         assert False
        # else:
        self.transform = None

    def __len__(self):
        return len(self._storage)

    def add(self, features, labels):
        # batch_size = ims.shape[0]

        # 1. organize features into different class
        # 2. extend storage & [len-_maxsize:]
        for i in range(self.num_classes):
            self._storage[i].extend(features.cpu().numpy()[np.where(labels.cpu().numpy()==i)])
            self._storage[i] = self._storage[i][len(self._storage[i])-self._maxsize:]

        # if self._next_idx >= len(self._storage):
        #     self._storage.extend(list(ims))
        # else:
        #     if batch_size + self._next_idx < self._maxsize:
        #         self._storage[self._next_idx:self._next_idx +
        #                       batch_size] = list(ims)
        #     else:
        #         split_idx = self._maxsize - self._next_idx
        #         self._storage[self._next_idx:] = list(ims)[:split_idx]
        #         self._storage[:batch_size - split_idx] = list(ims)[split_idx:]
        # self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize

    def _encode_sample(self, idxes, no_transform=False, downsample=False):
        features = []
        # pdb.set_trace()

        for i in range(len(idxes)):
            feat = self._storage[self.categories[i]][idxes[i]]

            # if self.dataset != "mnist":
                # if (self.transform is not None) and (not no_transform):
                #     im = im.transpose((1, 2, 0))
                #     im = np.array(self.transform(Image.fromarray(np.array(im))))

                # if downsample and (self.dataset in ["celeba", "object", "imagenet"]):
                #     im = im[:, ::4, ::4]

            # im = im * 255
            features.append(feat)
        return np.array(features)

    # def sample(self, batch_size, no_transform=False, downsample=False):
    def sample(self, neg_labels, no_transform=False, downsample=False):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        # idxes = [random.randint(0, len(self._storage) - 1)
        #          for _ in range(batch_size)]
        # batch_size = len(neg_labels)
        # self.categories = [random.randint(0, self.num_classes - 1)
        #          for _ in range(batch_size)]
        self.categories = neg_labels
        idxes = [random.randint(0, len(self._storage[cate]) - 1)
                 for cate in self.categories]
        # pdb.set_trace()
        return self._encode_sample(idxes, no_transform=no_transform, downsample=downsample), np.array(self.categories)

    # def set_elms(self, data, idxes):
    #     if len(self._storage) < self._maxsize:
    #         self.add(data)
    #     else:
    #         for i, ix in enumerate(idxes):
    #             self._storage[ix] = data[i]


class ReservoirBuffer(object):
    def __init__(self, size, transform, dataset):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.n = 0

        def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
            color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            color_distort = transforms.Compose([
                rnd_color_jitter,
                rnd_gray])
            return color_distort

        if dataset == "cifar10":
            im_size = 32
        elif dataset == "continual":
            im_size = 64
        elif dataset == "celeba":
            im_size = 128
        elif dataset == "object":
            im_size = 128
        elif dataset == "mnist":
            im_size = 28
        elif dataset == "moving_mnist":
            im_size = 28
        elif dataset == "imagenet":
            im_size = 128
        elif dataset == "lsun":
            im_size = 128
        elif dataset == "stl":
            im_size = 48
        else:
            assert False

        color_transform = get_color_distortion(0.5)
        self.dataset = dataset

        if transform:
            if dataset == "cifar10":
                color_transform = get_color_distortion(1.0)
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
                # self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.03, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])
            elif dataset == "continual":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])
            elif dataset == "celeba":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])
            elif dataset == "imagenet":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=11), transforms.ToTensor()])
            elif dataset == "lsun":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=5), transforms.ToTensor()])
            elif dataset == "stl":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.04, 1.0)), transforms.RandomHorizontalFlip(), color_transform, GaussianBlur(kernel_size=11), transforms.ToTensor()])
            elif dataset == "object":
                self.transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor()])
            elif dataset == "mnist":
                self.transform = None
            elif dataset == "moving_mnist":
                self.transform = None
            else:
                assert False
        else:
            self.transform = None

    def __len__(self):
        return len(self._storage)

    def add(self, ims):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
            self.n = self.n + ims.shape[0]
        else:
            for im in ims:
                self.n = self.n + 1
                ix = random.randint(0, self.n - 1)

                if ix < len(self._storage):
                    self._storage[ix] = im

        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize


    def _encode_sample(self, idxes, no_transform=False, downsample=False):
        ims = []
        for i in idxes:
            im = self._storage[i]

            if self.dataset != "mnist":
                if (self.transform is not None) and (not no_transform):
                    im = im.transpose((1, 2, 0))
                    im = np.array(self.transform(Image.fromarray(im)))

                # if downsample and (self.dataset in ["celeba", "object", "imagenet"]):
                #     im = im[:, ::4, ::4]

            im = im * 255

            ims.append(im)
        return np.array(ims)

    def sample(self, batch_size, no_transform=False, downsample=False):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes, no_transform=no_transform, downsample=downsample), idxes


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count