import os
import torchvision
import random
import numpy as np
import torch
from torch.optim import lr_scheduler
from PIL import Image

from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize,Grayscale
from data.imagecrop import FusionRandomCrop

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def get_scheduler(optimizer, args):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args['sheduler']['lr_policy'] == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args['n_epoch'] + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args['sheduler']['lr_policy'] == 'step':
        step_size = args['n_epoch']//args['sheduler']['n_steps']
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args['sheduler']['gamma'])
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def get_paths_from_mat(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_mat_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img

# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14

# augmentations for images
def transform_augment(img, split='val', min_max=(0, 1), res=256):
    hflip = torchvision.transforms.RandomHorizontalFlip()
    rcrop = torchvision.transforms.RandomCrop(size=res)
    resize = torchvision.transforms.Resize(size=res)
    totensor = torchvision.transforms.ToTensor()
    if split == 'train':
        if img.size[1] < res:
            img = resize(img)
        elif img.size[1] > res:
            img = rcrop(img)
        else:
            img=img
        img = hflip(img)
    img = totensor(img)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img


def transform_augment_cd(img, split='val', min_max=(0, 1)):
    totensor = torchvision.transforms.ToTensor()
    img = totensor(img)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.tif', '.bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        FusionRandomCrop(crop_size),
    ])


def train_vis_ir_transform():
    return Compose([
        Grayscale(num_output_channels=3),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        Grayscale(),
        ToTensor()
    ])