import matplotlib
import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(n1, n2, sharex=True, sharey=True)

    if n1*n2 > 1:
        if stacked:
            img = img[0]
            if target_channel > 0:
                order = [target_channel] + [i for i in range(len(img.shape)) if i != target_channel]
                img.transpose(tuple(order))

        
        for i in range(num_row):
            for j in range(num_col):
                plot_index = i*num_col + j
                ## TODO: use ax.plot() to plot everything on the subplots
                if isinstance(img[plot_index], torch.Tensor):
                    img_ = img[plot_index].data.cpu().numpy()
                else:
                    img_ = img[plot_index]

                if num_row == 1:
                    ax[j].imshow(img_)
                elif num_col == 1:
                    ax[i].imshow(img_)
                else:
                    
                    if order=='horizontal':
                        n1, n2 = j, i
                    else:
                        n1, n2 = i, j
                        
                    ax[n1, n2].imshow(img_)

    else:
        if isinstance(img[0], torch.Tensor):
            img_ = img[0].data.cpu().numpy()
        else:
            img_ = img[0]        
        ax.imshow(img_)
    if colormap is not None:
        plt.set_cmap(colormap)
    if flgHold:
        plt.show()
