from torchvision import transforms
import torch
from utils.superpixel_slic import superpixel
import numpy as np
from torch_scatter import scatter_mean, scatter_std

def transform_train(img, img_dim, **kwargs):

    # With superpixel
    if kwargs:
        img_transform_init = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        img_tensor = img_transform_init(img)
        super_pixel = kwargs['super_pixel']
        data = superpixel(img_tensor, super_pixel)
        normalize_transform = transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                                   std = (0.229, 0.224, 0.225))
        final_transform = transforms.Compose([transforms.Resize((img_dim, img_dim)),
                                              transforms.Normalize(mean = (0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                                                                   std = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225))])
        img = normalize_transform(data.img)
        data.img_super = final_transform(data.img_super)
        data.x = scatter_mean(img.view(img.shape[1]*img.shape[2], img.shape[0]),
                              data.seg.view(img.shape[1]*img.shape[2]), dim=0)
        data.x_std = scatter_std(img.view(img.shape[1]*img.shape[2], img.shape[0]),
                                 data.seg.view(img.shape[1]*img.shape[2]), dim=0)
        data.x = torch.cat([data.x, data.x_std], dim=1)
        super_index, super_counts = torch.unique(data.seg, return_counts=True)
        data.pos = torch.cat([data.pos.int(), super_counts.unsqueeze(1)], dim=1)
        if data.pos.shape[0] < super_pixel['segments']:
            pos_pad = torch.zeros(super_pixel['segments']-data.pos.shape[0], 3)
            x_pad = torch.zeros(super_pixel['segments']-data.x.shape[0], 6)
            data.pos = torch.cat((data.pos, pos_pad), dim=0)
            data.x = torch.cat((data.x, x_pad), dim=0)

        return data

    # No superpixel
    img_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.Resize((img_dim, img_dim)), transforms.ToTensor(),
         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    trans_img = img_transform(img)
    return trans_img


def transform_val(img, img_dim, **kwargs):

    if kwargs:
        img_transform_init = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = img_transform_init(img)
        super_pixel = kwargs['super_pixel']
        data = superpixel(img_tensor, super_pixel)
        normalize_transform = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                   std=(0.229, 0.224, 0.225))
        final_transform = transforms.Compose([transforms.Resize((img_dim, img_dim)),
                                              transforms.Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                                                                   std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225))])
        img = normalize_transform(data.img)
        data.img_super = final_transform(data.img_super)
        data.x = scatter_mean(img.view(img.shape[1] * img.shape[2], img.shape[0]),
                              data.seg.view(img.shape[1] * img.shape[2]), dim=0)
        data.x_std = scatter_std(img.view(img.shape[1] * img.shape[2], img.shape[0]),
                                 data.seg.view(img.shape[1] * img.shape[2]), dim=0)
        data.x = torch.cat([data.x, data.x_std], dim=1)
        super_index, super_counts = torch.unique(data.seg, return_counts=True)
        data.pos = torch.cat([data.pos.int(), super_counts.unsqueeze(1)], dim=1)
        if data.pos.shape[0] < super_pixel['segments']:
            pos_pad = torch.zeros(super_pixel['segments'] - data.pos.shape[0], 3)
            x_pad = torch.zeros(super_pixel['segments'] - data.x.shape[0], 6)
            data.pos = torch.cat((data.pos, pos_pad), dim=0)
            data.x = torch.cat((data.x, x_pad), dim=0)
        return data

    img_transform = transforms.Compose(
        [transforms.Resize((img_dim, img_dim)), transforms.ToTensor(),
         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    trans_img = img_transform(img)
    return trans_img