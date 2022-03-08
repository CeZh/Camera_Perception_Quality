from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from fast_slic.avx2 import SlicAvx2
import numpy as np
import torch
from torch_scatter import scatter_mean, scatter_std
from torch_geometric.data import Data
def superpixel(img, super_pixel, view=False):
    # fast SLIC superpixel
    img = img.permute(1, 2, 0)
    h, w, c = img.size()
    slic = SlicAvx2(num_components=super_pixel['segments'], compactness=10)

    img_numpy = (img*255).numpy().astype(np.uint8)
    segments_numpy = slic.iterate(img_numpy.copy(order='C'))
    segments = torch.from_numpy(segments_numpy).to(torch.int64)
    x = scatter_mean(img.view(h * w, c), segments.view(h * w), dim=0)
    pos_y = torch.arange(h, dtype=torch.float)
    pos_y = pos_y.view(-1, 1).repeat(1, w).view(h * w)
    pos_x = torch.arange(w, dtype=torch.float)
    pos_x = pos_x.view(1, -1).repeat(h, 1).view(h * w)
    pos = torch.stack([pos_x, pos_y], dim=-1)
    pos = scatter_mean(pos, segments.view(h * w), dim=0)
    super_img = x[segments]
    img_super = torch.cat((img, super_img), dim=2)
    data = Data(x=x, pos=pos, seg=segments, img=img.permute(2, 0, 1), img_super=img_super.permute(2, 0, 1))
    # segments = slic(img, n_segments=super_pixel['segments'], sigma=super_pixel['sigma'])
    if view:
        img = mark_boundaries(img, segments)
        plt.imshow(img)
        plt.show()
    return data