from math import floor, sqrt, ceil
from typing import Union
import torch
from torch import Tensor
from numpy import ndarray
import matplotlib.pyplot as plt


def visualize_avg_feature_map(feature: Union[Tensor,ndarray],
                              show: bool = True) -> None:
    '''
    :param feature: feature map with shape (B,N,H,C) to visualize
    :param show: show visualize or not
    '''

    if feature.shape == 4:
        feature = feature.squeeze(0)
        feature_map = torch.mean(feature, 0)
        feature_map = feature_map.detach().to('cpu').numpy()
        plt.figure(figsize=(10, 10))
        plt.title(f"The avarage feature map of layer ")
        plt.imshow(feature_map)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        if show:
            plt.show()


def visualize_all_kernal(feat_maps: Union[Tensor,ndarray],
                              show: bool = True):
    feat_maps = feat_maps.cpu().numpy()
    _, c, _, _ = feat_maps.shape
    nrows = floor(sqrt(c))
    ncols = ceil(c / nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        if idx < c:
            ax.imshow(feat_maps[0, idx])
        ax.set_xticks([])
        ax.set_yticks([])
    if show:
        plt.show()
