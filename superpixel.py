# from skimage import segmentation
from fast_slic import Slic
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

from random import randrange
import cv2
import os
from itertools import product, repeat
import matplotlib.pyplot as plt
import kornia as K
from utils import torch2numpy
# @timeit
def decode_sp(sp, lp):
    lp_ = torch2numpy(lp)
    sp_ = torch2numpy(sp)
    decoded_sp = []
    for (sp_i, lp_i) in zip(sp_, lp_):
        sp_i = sp_i.reshape(-1)
        lp_i = lp_i[lp_i > -1]
        lp_i = np.concatenate((np.zeros(1, ), lp_i), axis=0).astype('int')
        lp_i = np.cumsum(lp_i)
        decoded_sp_i = []
        for i in range(1, len(lp_i)):
            decoded_sp_i.append(sp_i[lp_i[i-1]:lp_i[i]])
        decoded_sp.append(decoded_sp_i)
    return decoded_sp
    
def slic(im, num_superpixels, compactness, mode):
    if mode == "slow":
            labels = segmentation.slic(im, compactness=compactness, n_segments=num_superpixels, max_iter=10)
    elif mode == "fast":
        slic = Slic(num_components=num_superpixels, compactness=compactness)
        # labels = slic.iterate(im.data.cpu().numpy()) # Cluster Map
        im = np.asarray(im, order='C')
        labels = slic.iterate(im) # Cluster Map
    else:
        raise("Undefined method")
    return labels

def vectorize_labels(labels):
    labels = labels.reshape(-1)
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])
    return l_inds

# @timeit
def extract_unsueprvised_segments(img, pool, num_superpixels=1000, compactness=100, mode="fast"):
    # label_index = pool.map(slic_vectorize, img)
    if pool is not None:
        label_index = pool.starmap(slic_vectorize, zip(img, repeat(num_superpixels), repeat(compactness)))
    else:
        label_index = [slic_vectorize(im, num_superpixels, compactness) for im in img]

    '''
    label_index = []
    for im in img:
        
        label_index.append(l_inds)
    '''
    sp_labels = [s for _,s in label_index]
    labels = [l for l,_ in label_index]
    return labels, sp_labels

def slic_vectorize(im, num_superpixels, compactness, mode="fast"):
    # num_superpixels += randrange(1800)-900
    # num_superpixels = 300
    # num_superpixels += randrange(200)-100
    # compactness += randrange(80)-40
    labels = slic(im, num_superpixels=num_superpixels, compactness=compactness, mode=mode)
    return [vectorize_labels(labels), labels]

def smooth_prediction_single(sp_k, label_k, num_class):
    label_smoothed = np.zeros_like(label_k).reshape(-1)
    num_sp = np.zeros((num_class, 1), np.int)
    num = len(sp_k)
    label_k = label_k.reshape(-1)
    for i in range(num):
        labels_per_sp = label_k[sp_k[i]]
        u_labels_per_sp = np.unique(labels_per_sp)
        num_unique = u_labels_per_sp.shape[0]
        if num_unique == 1:
            class_id = u_labels_per_sp[0]
        else:
            hist = np.histogram(labels_per_sp, bins=num_unique)[0]
            class_id = u_labels_per_sp[np.argmax(hist)]
        label_smoothed[sp_k[i]] = class_id
        num_sp[int(class_id)] += 1
    return (label_smoothed, num_sp)

# @timeit
def smooth_prediction(label, sp_index, num_class, pool=None):
    if pool is None:
        num_sp = np.zeros((num_class, 1), np.int)
        label_smoothed = label.clone().detach().reshape(label.shape[0], -1)
        for k, (sp_k, label_k) in enumerate(zip(sp_index, label)):
            num = len(sp_k)
            label_k = label_k.reshape(-1)
            for i in range(num):
                labels_per_sp = label_k[sp_k[i]]
                u_labels_per_sp = labels_per_sp.unique()
                num_unique = u_labels_per_sp.shape[0]
                if num_unique == 1:
                    class_id = u_labels_per_sp[0]
                else:
                    hist = torch.histc(labels_per_sp, bins=num_unique)
                    class_id = u_labels_per_sp[torch.argmax(hist)]
                label_smoothed[k][sp_k[i]] = class_id
                num_sp[int(class_id.detach().cpu().numpy())] += 1
            # label_smoothed = label_smoothed.astype('float16')
    else:
        output = pool.starmap(smooth_prediction_single, zip(sp_index, torch2numpy(label), repeat(num_class)))
        label_smoothed = torch.from_numpy(np.asarray([l for l,_ in output])).to(label.device)
        num_sp = [n for _,n in output]
        num_sp = np.asarray(num_sp).squeeze().sum(axis=0)
    return label_smoothed, num_sp

# @timeit
def extract_sp_index(sp, mode_sp='subMask', ignore_label=None, pool=None):        
    if pool is None:
        sp_label = []
        for superpixel in sp:
            sp_label.append(extarct_sp_index_single(superpixel,  mode=mode_sp, ignore_label=ignore_label))
        return sp_label
    else:
        if type(sp) is torch.Tensor:
            sp = torch2numpy(sp)
        return pool.starmap(extarct_sp_index_single, zip(sp, repeat(True), repeat(mode_sp), repeat(False), repeat(ignore_label)))

def extarct_sp_index_single(superpixel, flgParallel=False, mode='subMask', preprocess=False, ignore_label=None):
    if preprocess :
        superpixel = cv2.medianBlur(superpixel.astype('uint16'), 3)
    labels = superpixel.reshape(-1)
    indices = np.indices((superpixel.shape[0]*superpixel.shape[1], )).T[:, 0]
    l_pil = [] # label pixel index list
    if not flgParallel and torch.is_tensor(labels):
        u_labels = labels.unique()
        if ignore_label is not None:
            u_labels = u_labels[u_labels!=ignore_label]
        for u_lab in u_labels:
            l_pil.append(torch.where(labels == u_lab)[0])
    else:
        u_labels = np.unique(labels)
        if ignore_label is not None:
            u_labels = u_labels[u_labels!=ignore_label]
        for u_lab in u_labels:
            if mode=='subMask':
                labels_i = bwlabel(superpixel == u_lab).reshape(-1)
                u_labels_i = np.unique(labels_i)[1:]
                for u_lab_i in u_labels_i:
                    l_pil.append(indices[labels_i == u_lab_i]) # faster than np.where
                    # l_pil.append(np.where(labels_i == u_lab_i)[0])
            elif mode=='Mask':
                l_pil.append(np.where(labels == u_lab)[0])

    return l_pil    

def bwlabel(mask):
    return cv2.connectedComponents(mask.astype('uint8'))[1]

def unique_label(superpixel, min_size=0):
    if type(superpixel) is torch.Tensor:
        label_cnt = 1
        unique_superpixel = torch.zeros_like(superpixel, dtype=torch.int32)
        
        # We could have zero index, for we utilize approximate arg max, not the definite and exact one
        u_labels = torch.unique(superpixel)
        for u_lab in u_labels:
            labels_i = torch.from_numpy(bwlabel(torch2numpy(superpixel == u_lab))).cuda()
            
            # Here, we just look for integer index and discard zeros index
            u_labels_i = torch.unique(labels_i[labels_i>0], return_counts=True)
            u_labels_i = u_labels_i[0][u_labels_i[1] > min_size]
            for u_lab_i in u_labels_i:
                unique_superpixel[labels_i == u_lab_i] = label_cnt
                label_cnt += 1
        return unique_superpixel
    else:
        label_cnt = 1
        unique_superpixel = np.zeros_like(superpixel, dtype='int32')
        u_labels = np.unique(superpixel)
        for u_lab in u_labels:
            labels_i = bwlabel(superpixel == u_lab)
            u_labels_i = np.unique(labels_i[labels_i>0], return_counts=True)
            u_labels_i = u_labels_i[0][u_labels_i[1] > min_size]
            for u_lab_i in u_labels_i:
                unique_superpixel[labels_i == u_lab_i] = label_cnt
                label_cnt += 1
        return unique_superpixel
    
def superpixel_refinment(label_index, targets, num_class = 80):
    num_sp = np.zeros((num_class, 1), np.int)
    img_target = targets.detach().cpu().numpy().copy()
    # img_target = targets.copy()
    # target = Variable(torch.zeros((target.shape[0], target.shape[1]*target.shape[2])))
    target = np.zeros((targets.shape[0], targets.shape[1]*targets.shape[2]))

    # superpixel refinement
    # TODO: use Torch Variable instead of numpy for faster calculation
    for k, (l_inds, im_target) in enumerate(zip(label_index, img_target)):
        im_target = im_target.reshape(-1)
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
            im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
            num_sp[int(u_labels_per_sp[ np.argmax( hist ) ])] += 1
        # target[k] = Variable(torch.from_numpy(im_target.astype('float16')))
        target[k] = im_target.astype('float16')
    return target, num_sp

# @timeit
def superpixel_refinment_old(label_index, targets, num_class = 80):
    num_sp = np.zeros((num_class, 1), np.int)
    img_target = targets.detach().cpu().numpy().copy()
    # img_target = targets.copy()
    # target = Variable(torch.zeros((target.shape[0], target.shape[1]*target.shape[2])))
    target = np.zeros((targets.shape[0], targets.shape[1]*targets.shape[2]))

    # superpixel refinement
    # TODO: use Torch Variable instead of numpy for faster calculation
    for k, (l_inds, im_target) in enumerate(zip(label_index, img_target)):
        im_target = im_target.reshape(-1)
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
            im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
            num_sp[int(u_labels_per_sp[ np.argmax( hist ) ])] += 1
        # target[k] = Variable(torch.from_numpy(im_target.astype('float16')))
        target[k] = im_target.astype('float16')
        
    
    return target, num_sp

def filter_sp(sp_index, MIN_SP_AREA):
    sp = []
    for sp_i in sp_index:
        sp_j = []
        for sp_ij in sp_i:
            if len(sp_ij) > MIN_SP_AREA:
                sp_j.append(sp_ij)
        sp.append(sp_j)
    return sp
    
def superpixel(sp, lp, sp_image, mode_sp, pool, MIN_SP_AREA):
    
    if sp is not None:
        if not(sp_image):
            sp = decode_sp(sp, lp)
        else:
            # discard invalid superpixels here to speed up training
            sp = extract_sp_index(sp,  mode_sp=mode_sp, pool=pool, ignore_label=0)
            sp = filter_sp(sp, MIN_SP_AREA)
    else:
        sp=None
    return sp