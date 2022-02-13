import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from .box_analysis import box2poly
from .utils import isClass
try:
    import torch
except:
    pass

try:
    matplotlib.use('TkAgg')
except:
    pass

def lax(num_row, num_col, *img, flgHold=True, stacked=False, target_channel=0, order='vertical', colormap=None):

    if order=='horizontal':
        n2 = num_row
        n1 = num_col
    else:
        n1 = num_row
        n2 = num_col

    n = len(img)

    if n == n1*n2:
        coef = 1
        meta = [None]

    elif n == 2*n1*n2:
        coef = 2
        if isClass(img[1], str):
            meta = ['title']
        elif isClass(img[1], list):
            meta = ['plot']

    elif n == 3*n1*n2:
        coef = 3
        if isClass(img[1],str):
            meta = ['title']
        elif isClass(img[1], list):
            meta = ['plot']
        if isClass(img[2], str):
            meta.append(['title'])
        elif isClass(img[2], list):
            meta.append(['plot'])

    fig, ax = plt.subplots(n1, n2, sharex=True, sharey=True)

    if coef > 1:
        if stacked:
            img = img[0]
            if target_channel > 0:
                order = [target_channel] + [i for i in range(len(img.shape)) if i != target_channel]
                img.transpose(tuple(order))

        for i in range(num_row):
            for j in range(num_col):
                plot_index = i*num_col + j
                ## TODO: use ax.plot() to plot everything on the subplots
                if isinstance(img[coef*plot_index], torch.Tensor):
                    img_ = img[plot_index].data.cpu().numpy()
                else:
                    img_ = img[coef*plot_index]

                if num_row == 1:
                    ax[j].imshow(img_)
                    ax_ = ax[j]
                elif num_col == 1:
                    ax[i].imshow(img_)
                    ax_ = ax[i]
                else:

                    if order=='horizontal':
                        n1, n2 = j, i
                    else:
                        n1, n2 = i, j
                        
                    ax[n1, n2].imshow(img_)
                    ax_ = ax[n1, n2]

                for _i, meta_i in enumerate(meta):
                    if meta_i == 'title':
                        ax_.set_title(img[coef*plot_index+_i+1])
                    else: #'plot'
                        for pts_i in box2poly(img[coef*plot_index+_i+1]):
                            pts_i = np.array(pts_i)
                            x, y = pts_i[:, 0], pts_i[:, 1]
                            row, col = img_.shape[0], img_.shape[1]
                            x = np.maximum(np.minimum(x, col-1), 0)
                            y = np.maximum(np.minimum(y, row-1), 0)
                            ax_.plot(x, y)
    else:   
        if isinstance(img[0], torch.Tensor):
            img_ = img[0].data.cpu().numpy()
        else:
            img_ = img[0]        
        ax.imshow(img_)
        for _i, meta_i in enumerate(meta):
            if meta_i == 'title':
                ax.title = img[coef*plot_index+_i]
            else: #'plot'
                ax.plot(img[coef*plot_index+_i])

    if colormap is not None:
        plt.set_cmap(colormap)
    if flgHold:
        plt.show()
