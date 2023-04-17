from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt


def show_img(im: Union[np.ndarray, torch.Tensor, Image.Image], fig: plt.figure = None,
             transpose: bool = False, cvt_color: bool = False):
    # plt.axis('off')
    im = np.asarray(im)

    if transpose:
        im = im.transpose(1, 2, 0)

    if cvt_color:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if fig:
        fig.imshow(im)
    else:
        plt.imshow(im)
        plt.show()


def show_imgs(*imgs, row=1,
              transpose: bool = False, cvt_color: bool = False):
    # 创建图表网格, 其中row和col指定图表的行数和列数
    fig, axs = plt.subplots(nrows=row, ncols=int(np.ceil(len(imgs) / row)))

    # 循环每个子图, 并在其中展示图像
    for im, ax in zip(imgs, axs.flat):
        show_img(im, ax, transpose, cvt_color)
        ax.axis('off')
    plt.show()
