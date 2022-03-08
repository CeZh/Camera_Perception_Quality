import json, os
import torch
import cv2
import numpy as np
from utils.lds_utils import get_lds_kernel_window
from scipy.ndimage import convolve1d
from utils.image_compression import transform_train, transform_val
# for image quality assessment
import csv
from PIL import Image
def register_dataset(name):
    if name == 'bdd100k':
        name = BDD_Dataset
    elif name == 'nuscene':
        name = NuScene_Dataset
    elif name == 'koniq-10k':
        name = Koniq10K_Dataset
    elif name == 'kitti':
        name = KITTI_Dataset
    return name

@register_dataset
class NuScene_Dataset():
    def __init__(self, image_root, label_root, mode, **kwargs): # unbalance_lds, lds_method, lds_ks, lds_sigma, max_target):

        # data root
        self.image_root = image_root
        self.label_root = label_root

        # import datat
        self.mode = mode
        with open(label_root) as f:
            image_annotation = json.load(f)
        self.image_annotation = image_annotation
        perceptual_quality_regress = []

        # check LDS or Superpixel
        self.kwargs = kwargs['extra']

        # LDS
        if self.kwargs and self.kwargs['dataset_parameters']['use_lds']:
            parameters = kwargs['extra']['dataset_parameters']
            for i in range(len(self.image_annotation)):
                perceptual_quality_regress.append(self.image_annotation[i]['perceptual_score'])
            self.weights = self._weights_generation(perceptual_quality_regress,
                                                    reweight_method=parameters['lds_method'],
                                                    lds_ks=parameters['lds_ks'],
                                                    lds_sigma=parameters['lds_sigma'],
                                                    max_target=parameters['max_target'])

        # Superpixel
        if self.kwargs and self.kwargs['model_parameters']['use_superpixel']:
            self.super_pixel = self.kwargs['superpixel_parameters']
    def __len__(self):
        return len(self.image_annotation)
    def __getitem__(self, idx):
        file_name = self.image_annotation[idx]['image_id']
        image_name = file_name
        img = self.pil_loader(os.path.join(self.image_root, image_name))

        # train data for
        if self.mode == 'train':
            if self.kwargs and self.kwargs['model_parameters']['use_superpixel']:
                img = transform_train(img,
                                      img_dim = self.kwargs['model_parameters']['img_dim'],
                                      super_pixel=self.super_pixel)
            else:
                img = transform_train(img,
                                      img_dim = self.kwargs['model_parameters']['img_dim'],)
        elif self.mode == 'val':
            if self.kwargs and self.kwargs['model_parameters']['use_superpixel']:
                img = transform_val(img,
                                    img_dim = self.kwargs['model_parameters']['img_dim'],
                                    super_pixel=self.super_pixel)
            else:
                img = transform_val(img, img_dim = self.kwargs['model_parameters']['img_dim'])
        perceptual_quality = self.image_annotation[idx]['perceptual_score']

        if self.kwargs and self.kwargs['dataset_parameters']['use_lds']:
            weights = self.weights[idx]
            return img, perceptual_quality, weights, image_name
        return img, perceptual_quality, image_name
    def _weights_generation(self, labels, reweight_method, lds_ks, lds_sigma, max_target):
        reweight = reweight_method
        lds = True
        lds_kernel = 'gaussian'

        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        # mbr
        for label in labels:
            value_dict[min((max_target - 1), int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            print('None')


        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)

            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]

        return weights
    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

@register_dataset
class KITTI_Dataset():
    def __init__(self, image_root, label_root, mode, **kwargs): # unbalance_lds, lds_method, lds_ks, lds_sigma, max_target):

        # data root
        self.image_root = image_root
        self.label_root = label_root

        # import datat
        self.mode = mode
        with open(label_root) as f:
            image_annotation = json.load(f)
        self.image_annotation = image_annotation
        perceptual_quality_regress = []

        # check LDS or Superpixel
        self.kwargs = kwargs['extra']

        # LDS
        if self.kwargs and self.kwargs['dataset_parameters']['use_lds']:
            parameters = kwargs['extra']['dataset_parameters']
            for i in range(len(self.image_annotation)):
                perceptual_quality_regress.append(self.image_annotation[i]['perceptual_score'])
            self.weights = self._weights_generation(perceptual_quality_regress,
                                                    reweight_method=parameters['lds_method'],
                                                    lds_ks=parameters['lds_ks'],
                                                    lds_sigma=parameters['lds_sigma'],
                                                    max_target=parameters['max_target'])

        # Superpixel
        if self.kwargs and self.kwargs['model_parameters']['use_superpixel']:
            self.super_pixel = self.kwargs['superpixel_parameters']
    def __len__(self):
        return len(self.image_annotation)
    def __getitem__(self, idx):
        file_name = self.image_annotation[idx]['image_id']
        image_name = file_name
        img = self.pil_loader(os.path.join(self.image_root, image_name))

        # train data for
        if self.mode == 'train':
            if self.kwargs and self.kwargs['model_parameters']['use_superpixel']:
                img = transform_train(img,
                                      img_dim = self.kwargs['model_parameters']['img_dim'],
                                      super_pixel=self.super_pixel)
            else:
                img = transform_train(img,
                                      img_dim = self.kwargs['model_parameters']['img_dim'],)
        elif self.mode == 'val':
            if self.kwargs and self.kwargs['model_parameters']['use_superpixel']:
                img = transform_val(img,
                                    img_dim = self.kwargs['model_parameters']['img_dim'],
                                    super_pixel=self.super_pixel)
            else:
                img = transform_val(img, img_dim = self.kwargs['model_parameters']['img_dim'])
        perceptual_quality = self.image_annotation[idx]['perceptual_score']

        if self.kwargs and self.kwargs['dataset_parameters']['use_lds']:
            weights = self.weights[idx]
            return img, perceptual_quality, weights, image_name
        return img, perceptual_quality, image_name
    def _weights_generation(self, labels, reweight_method, lds_ks, lds_sigma, max_target):
        reweight = reweight_method
        lds = True
        lds_kernel = 'gaussian'

        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        # mbr
        for label in labels:
            value_dict[min((max_target - 1), int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            print('None')


        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)

            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]

        return weights
    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

@register_dataset
class BDD_Dataset():
    def __init__(self, image_root, label_root, mode, **kwargs): # unbalance_lds, lds_method, lds_ks, lds_sigma, max_target):

        # data root
        self.image_root = image_root
        self.label_root = label_root

        # import datat
        self.mode = mode
        with open(label_root) as f:
            image_annotation = json.load(f)
        self.image_annotation = image_annotation
        perceptual_quality_regress = []

        # check LDS or Superpixel
        self.kwargs = kwargs['extra']

        # LDS
        if self.kwargs and self.kwargs['dataset_parameters']['use_lds']:
            parameters = kwargs['extra']['dataset_parameters']
            for i in range(len(self.image_annotation)):
                perceptual_quality_regress.append(self.image_annotation[i]['perceptual_score'])
            self.weights = self._weights_generation(perceptual_quality_regress,
                                                    reweight_method=parameters['lds_method'],
                                                    lds_ks=parameters['lds_ks'],
                                                    lds_sigma=parameters['lds_sigma'],
                                                    max_target=parameters['max_target'])

        # Superpixel
        if self.kwargs and self.kwargs['model_parameters']['use_superpixel']:
            self.super_pixel = self.kwargs['superpixel_parameters']
    def __len__(self):
        return len(self.image_annotation)
    def __getitem__(self, idx):
        file_name = self.image_annotation[idx]['image_id']
        image_name = file_name
        img = self.pil_loader(os.path.join(self.image_root, image_name))

        # train data for
        if self.mode == 'train':
            if self.kwargs and self.kwargs['model_parameters']['use_superpixel']:
                img = transform_train(img,
                                      img_dim = self.kwargs['model_parameters']['img_dim'],
                                      super_pixel=self.super_pixel)
            else:
                img = transform_train(img,
                                      img_dim = self.kwargs['model_parameters']['img_dim'],)
        elif self.mode == 'val':
            if self.kwargs and self.kwargs['model_parameters']['use_superpixel']:
                img = transform_val(img,
                                    img_dim = self.kwargs['model_parameters']['img_dim'],
                                    super_pixel=self.super_pixel)
            else:
                img = transform_val(img, img_dim = self.kwargs['model_parameters']['img_dim'])
        perceptual_quality = self.image_annotation[idx]['perceptual_score']

        if self.kwargs and self.kwargs['dataset_parameters']['use_lds']:
            weights = self.weights[idx]
            return img, perceptual_quality, weights, image_name
        return img, perceptual_quality, image_name
    def _weights_generation(self, labels, reweight_method, lds_ks, lds_sigma, max_target):
        reweight = reweight_method
        lds = True
        lds_kernel = 'gaussian'

        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        # mbr
        for label in labels:
            value_dict[min((max_target - 1), int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            print('None')


        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)

            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]

        return weights
    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

@register_dataset
class Koniq10K_Dataset():
    def __init__(self, image_root, label_root, mode, index, patch_num, **kwargs):
        self.image_root = image_root
        self.label_root = label_root
        self.mode = mode
        self.kwargs = kwargs
        if kwargs:
            self.super_pixel = kwargs['super_pixel']
        mos_all = []
        imgname = []
        with open(os.path.join(self.label_root, 'koniq10k_scores_and_distributions.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['sharpness'])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(image_root, imgname[item]), mos_all[item]))
        self.samples = sample

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.pil_loader(path)
        if self.mode == 'train':
            if self.kwargs:
                image_sample = transform_train(sample, super_pixel=self.super_pixel)
            else:
                image_sample = transform_train(sample)
        elif self.mode == 'val':
            if self.kwargs:
                image_sample = transform_val(sample, super_pixel=self.super_pixel)
            else:
                image_sample = transform_train(sample)

        return image_sample, target

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

def collate_bdd(data):
    if type(data[0][0]) == torch.Tensor:
        if len(data[0]) == 3:
            img2stack = np.stack(d[0] for d in data)
            label2stack = np.stack(d[1] for d in data)
            name2stack = np.stack(d[2] for d in data)
            data = {'image': img2stack,
                    'labels': label2stack,
                    'name': name2stack}
    else:
        if len(data[0]) == 3:
            img2stack = np.stack(d[0].img_super for d in data)
            pos2stack = np.stack(d[0].pos for d in data)
            seg2stack = np.stack(d[0].seg for d in data)
            superx2stack = np.stack(d[0].x for d in data)
            label2stack = np.stack(d[1] for d in data)
            data = {'image': img2stack,
                    'superpixel': superx2stack,
                    'superpos': pos2stack,
                    'superseg': seg2stack,
                    'labels': label2stack}
        else:
            img2stack = np.stack(d[0].img_super for d in data)
            pos2stack = np.stack(d[0].pos for d in data)
            seg2stack = np.stack(d[0].seg for d in data)
            superx2stack = np.stack(d[0].x for d in data)
            label2stack = np.stack(d[1] for d in data)
            weight2stack = np.stack(d[2] for d in data)
            name2stack = np.stack(d[3] for d in data)
            data = {'image': img2stack,
                    'superpixel': superx2stack,
                    'superpos': pos2stack,
                    'superseg': seg2stack,
                    'labels': label2stack,
                    'weights': weight2stack,
                    'names': name2stack}
    return data

def collate_koniq(data):
    if type(data[0][0]) == torch.Tensor:
        img2stack = np.stack(d[0] for d in data)
        label2stack = np.stack(d[1] for d in data)
        data = {'image': img2stack,
                'labels': label2stack}
    else:
        img2stack = np.stack(d[0].img_super for d in data)
        pos2stack = np.stack(d[0].pos for d in data)
        seg2stack = np.stack(d[0].seg for d in data)
        superx2stack = np.stack(d[0].x for d in data)
        label2stack = np.stack(d[1] for d in data)
        data = {'image': img2stack,
                'superpixel': superx2stack,
                'superpos': pos2stack,
                'superseg': seg2stack,
                'labels': label2stack}

    return data

def collate_kitti(data):
    if type(data[0][0]) == torch.Tensor:
        if len(data[0]) == 3:
            img2stack = np.stack(d[0] for d in data)
            label2stack = np.stack(d[1] for d in data)
            name2stack = np.stack(d[2] for d in data)
            data = {'image': img2stack,
                    'labels': label2stack,
                    'name': name2stack}
    else:
        if len(data[0]) == 3:
            img2stack = np.stack(d[0].img_super for d in data)
            pos2stack = np.stack(d[0].pos for d in data)
            superx2stack = np.stack(d[0].x for d in data)
            label2stack = np.stack(d[1] for d in data)
            data = {'image': img2stack,
                    'superpixel': superx2stack,
                    'superpos': pos2stack,
                    'labels': label2stack}
        else:
            img2stack = np.stack(d[0].img_super for d in data)
            pos2stack = np.stack(d[0].pos for d in data)
            superx2stack = np.stack(d[0].x for d in data)
            label2stack = np.stack(d[1] for d in data)
            weight2stack = np.stack(d[2] for d in data)
            name2stack = np.stack(d[3] for d in data)
            data = {'image': img2stack,
                    'superpixel': superx2stack,
                    'superpos': pos2stack,
                    'labels': label2stack,
                    'weights': weight2stack,
                    'names': name2stack}
    return data

def collate_nuscene(data):
    if type(data[0][0]) == torch.Tensor:
        if len(data[0]) == 3:
            img2stack = np.stack(d[0] for d in data)
            label2stack = np.stack(d[1] for d in data)
            name2stack = np.stack(d[2] for d in data)
            data = {'image': img2stack,
                    'labels': label2stack,
                    'name': name2stack}
    else:
        if len(data[0]) == 3:
            img2stack = np.stack(d[0].img_super for d in data)
            pos2stack = np.stack(d[0].pos for d in data)
            seg2stack = np.stack(d[0].seg for d in data)
            superx2stack = np.stack(d[0].x for d in data)
            label2stack = np.stack(d[1] for d in data)
            data = {'image': img2stack,
                    'superpixel': superx2stack,
                    'superpos': pos2stack,
                    'superseg': seg2stack,
                    'labels': label2stack}
        else:
            img2stack = np.stack(d[0].img_super for d in data)
            pos2stack = np.stack(d[0].pos for d in data)
            seg2stack = np.stack(d[0].seg for d in data)
            superx2stack = np.stack(d[0].x for d in data)
            label2stack = np.stack(d[1] for d in data)
            weight2stack = np.stack(d[2] for d in data)
            name2stack = np.stack(d[3] for d in data)
            data = {'image': img2stack,
                    'superpixel': superx2stack,
                    'superpos': pos2stack,
                    'superseg': seg2stack,
                    'labels': label2stack,
                    'weights': weight2stack,
                    'names': name2stack}
    return data