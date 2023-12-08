import os
import numpy as np
import torch
from PIL import Image
import torchvision
import cv2
import matplotlib.pyplot as plt
import utils
import torch.nn as nn
import torch.nn.functional as F
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# Inpyt image size
ISIZE = (800, 800)

# Imagenet statistics
imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])

## Anchor Box Specs
ratios = [0.5, 1, 2]
anchor_scales = [8, 16, 32]


def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], color=color,
                         fill=False, lw=3)


def show_corner_bbs(im, bbs):
    # im = np.asarray(im).astype(int).transpose(1,2,0)
    im = np.asarray(im).transpose(1, 2, 0)
    im = unnormalize(im)
    plt.imshow(im)
    for bb in bbs:
        plt.gca().add_patch(create_corner_rect(bb))


def normalize(im):
    # im = im.astype(np.float32)/255.
    im = im / 255.
    """Normalizes images with Imagenet stats."""
    return (im - imagenet_stats[0]) / imagenet_stats[1]


def unnormalize(im):
    im = im.astype(np.float32)
    """Normalizes images with Imagenet stats."""
    im = (im * imagenet_stats[1] + imagenet_stats[0]) * 255.
    im = im.astype(np.int)
    return im


def train_val_dataset(dataset, val_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def evaluate_feature_map(imgs, req_features):
    k = imgs.clone()
    for m in req_features:
        k = m(k)
    return k


def plot_feature_map(k, img_idx, row=64, col=8):
    fig = plt.figure(figsize=(20, 160))
    fig.tight_layout()
    for i in range(row):
        for j in range(col):
            ind = i * col + j
            ax = fig.add_subplot(row, col, ind + 1, xticks=[], yticks=[])
            ax.imshow(k[img_idx, ind, :, :].detach().cpu())
            ax.text(3, 6, ind, fontdict={'weight': 'bold', 'size': 16}, color="y")
    fig.suptitle("feature map")


def plot_imges_with_bboxes(imgs, bboxes, col=4):
    num_imgs = len(imgs)
    row = int(np.ceil(float(len(imgs)) / col).item())
    print("row: %s" % row)
    print("imgs: ", num_imgs)
    fig = plt.figure(figsize=(2 * col, 2 * row))
    fig.tight_layout()
    for i in range(row):
        for j in range(col):
            ind = i * col + j
            if ind < num_imgs:
                img = imgs[ind]
                bbs = bboxes[ind]
                img = np.asarray(img).transpose(1, 2, 0)
                img = unnormalize(img)
                ax = fig.add_subplot(row, col, ind + 1, xticks=[], yticks=[])
                ax.imshow(img)
                for bb in bbs:
                    plt.gca().add_patch(create_corner_rect(bb))
            # ax.text(3, 6, ind, fontdict={'weight': 'bold', 'size': 16}, color="y" )
    fig.suptitle("Images")


def pred_bbox_to_xywh(bbox, anchors):
    print("bbox: ", bbox.shape)
    print("anchors: ", anchors.shape)
    anc_height = anchors[:, 2] - anchors[:, 0]
    anc_width = anchors[:, 3] - anchors[:, 1]
    anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
    anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

    bbox_numpy = bbox.detach().cpu().data.numpy()
    dy = bbox_numpy[:, 0::4]
    dx = bbox_numpy[:, 1::4]
    dh = bbox_numpy[:, 2::4]
    dw = bbox_numpy[:, 3::4]
    ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
    ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
    h = np.exp(dh) * anc_height[:, np.newaxis]
    w = np.exp(dw) * anc_width[:, np.newaxis]

    roi = np.zeros(bbox_numpy.shape, dtype=h.dtype)
    roi[:, 0::4] = ctr_y - 0.5 * h
    roi[:, 1::4] = ctr_x - 0.5 * w
    roi[:, 2::4] = ctr_y + 0.5 * h
    roi[:, 3::4] = ctr_x + 0.5 * w

    return roi


def bbox_generation(images, targets, X_FM, Y_FM):
    global ratios
    global anchor_scales
    num_batch = len(images)
    X_IMG, Y_IMG = images[0].shape[1:]
    bbox_all = [item['boxes'] for item in targets]
    labels_all = [item['labels'] for item in targets]

    # imgs_torch_all = torch.stack([item for item in images])
    # if is_cuda:
    #    imgs_torch_all = imgs_torch_all.cuda()
    # k = imgs_torch_all.clone()
    # for m in req_features:
    #    k = m(k)
    # print(k.shape)

    sub_sampling_x = int(X_IMG / X_FM)
    sub_sampling_y = int(Y_IMG / Y_FM)
    # print(X_IMG, Y_IMG, X_FM, Y_FM, sub_sampling_x,sub_sampling_y)
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)

    ctr_x = np.arange(sub_sampling_x, (X_FM + 1) * sub_sampling_x, sub_sampling_x)
    ctr_y = np.arange(sub_sampling_y, (Y_FM + 1) * sub_sampling_y, sub_sampling_y)
    index = 0
    ctr = np.zeros((len(ctr_y) * len(ctr_y), 2), dtype=np.float32)
    for x in range(len(ctr_x)):
        for y in range(len(ctr_y)):
            ctr[index, 1] = ctr_x[x] - 8
            ctr[index, 0] = ctr_y[y] - 8
            index += 1

    anchors = np.zeros((X_FM * Y_FM * 9, 4))
    index = 0
    for ctr_y, ctr_x in ctr:
        for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
                h = sub_sampling_x * anchor_scales[j] * np.sqrt(ratios[i])
                w = sub_sampling_y * anchor_scales[j] * np.sqrt(1. / ratios[i])
                anchors[index, 0] = ctr_y - h / 2.
                anchors[index, 1] = ctr_x - w / 2.
                anchors[index, 2] = ctr_y + h / 2.
                anchors[index, 3] = ctr_x + w / 2.
                index += 1
    # print(anchors.shape)

    index_inside = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= Y_IMG) &
        (anchors[:, 3] <= X_IMG)
    )[0]
    # print(index_inside.shape)

    label = np.empty((len(index_inside),), dtype=np.int32)
    label.fill(-1)
    valid_anchors = anchors[index_inside]
    # print(label.shape, valid_anchors.shape)
    # print(valid_anchors[0])

    ious_all = []
    for bx in bbox_all:
        ious = np.empty((len(label), bx.size()[0]), dtype=np.float32)
        ious.fill(0)
        for num1, i in enumerate(valid_anchors):
            ya1, xa1, ya2, xa2 = i
            anchor_area = (ya2 - ya1) * (xa2 - xa1)
            for num2, j in enumerate(bx):
                yb1, xb1, yb2, xb2 = j
                box_area = (yb2 - yb1) * (xb2 - xb1)
                inter_x1 = max([xb1, xa1])
                inter_y1 = max([yb1, ya1])
                inter_x2 = min([xb2, xa2])
                inter_y2 = min([yb2, ya2])
                if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                    iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                    iou = iter_area / (anchor_area + box_area - iter_area)
                else:
                    iou = 0.
                ious[num1, num2] = iou
        ious_all.append(ious)

    gt_argmax_ious_all = []
    gt_max_ious_all = []
    for ious_ in ious_all:
        gt_argmax_ious = ious_.argmax(axis=0)
        gt_max_ious = ious_[gt_argmax_ious, np.arange(ious_.shape[1])]
        gt_argmax_ious_all.append(gt_argmax_ious)
        gt_max_ious_all.append(gt_max_ious)
    # print(gt_argmax_ious_all)
    # print(gt_max_ious_all)

    argmax_ious_all = []
    max_ious_all = []
    for ious_ in ious_all:
        argmax_ious = ious_.argmax(axis=1)
        max_ious = ious_[np.arange(len(label)), argmax_ious]
        argmax_ious_all.append(argmax_ious)
        max_ious_all.append(max_ious)
    # print(argmax_ious_all)
    # print(max_ious_all)

    gt_argmax_ious_all = []
    for gt_max_ious_, ious_ in zip(gt_max_ious_all, ious_all):
        gt_argmax_ious = np.where(ious_ == gt_max_ious_)[0]
        gt_argmax_ious_all.append(gt_argmax_ious)
    # print(gt_argmax_ious_all)

    pos_iou_threshold = 0.7
    neg_iou_threshold = 0.3

    label_all = []
    for n in range(num_batch):
        l = copy.deepcopy(label)
        l[max_ious_all[n] < neg_iou_threshold] = 0
        l[gt_argmax_ious_all[n]] = 1
        l[max_ious_all[n] >= pos_iou_threshold] = 1
        label_all.append(l)
    # print ("label_all 0 and 1: ", sum(label_all[0]), sum(label_all[1]))

    pos_ratio = 0.5
    n_sample = 256
    n_pos = int(pos_ratio * n_sample)
    # print(n_pos)

    for n in range(num_batch):
        # print(np.sum((label_all[n] == 1)))
        pos_index = np.where(label_all[n] == 1)[0]
        # print(pos_index)
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label_all[n][disable_index] = -1
        # print(np.sum((label_all[n] == 1)))

        n_neg = n_sample - np.sum(label_all[n] == 1)
        neg_index = np.where(label_all[n] == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label_all[n][disable_index] = -1
        # print(np.sum((label_all[n] == 0)))

    max_iou_bbox_all = []
    # print(bbox_all)
    for n in range(num_batch):
        max_iou_bbox_all.append(bbox_all[n][argmax_ious_all[n]])
    # print(max_iou_bbox_all[0].shape, max_iou_bbox_all[0].shape)

    # Anchor box
    height = valid_anchors[:, 2] - valid_anchors[:, 0]
    width = valid_anchors[:, 3] - valid_anchors[:, 1]
    ctr_y = valid_anchors[:, 0] + 0.5 * height
    ctr_x = valid_anchors[:, 1] + 0.5 * width
    # Ground truth
    base_height_all = []
    base_width_all = []
    base_ctr_y_all = []
    base_ctr_x_all = []
    for n in range(num_batch):
        base_height = max_iou_bbox_all[n][:, 2] - max_iou_bbox_all[n][:, 0]
        base_width = max_iou_bbox_all[n][:, 3] - max_iou_bbox_all[n][:, 1]
        base_ctr_y = max_iou_bbox_all[n][:, 0] + 0.5 * base_height
        base_ctr_x = max_iou_bbox_all[n][:, 1] + 0.5 * base_width
        base_height_all.append(base_height)
        base_width_all.append(base_width)
        base_ctr_y_all.append(base_ctr_y)
        base_ctr_x_all.append(base_ctr_x)

    # print(width[2], base_width_all[0][2])

    # Prevent devide by 0
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    # d_{} calculatrion
    anchor_locs_all = []
    for n in range(num_batch):
        dy = (base_ctr_y_all[n].numpy() - ctr_y) / height
        dx = (base_ctr_x_all[n].numpy() - ctr_x) / width
        dh = np.log(base_height_all[n].numpy() / height)
        dw = np.log(base_width_all[n].numpy() / width)
        anchor_locs_all.append(np.vstack((dy, dx, dh, dw)).transpose())
    # print(anchor_locs_all[0][1], anchor_locs_all[0].shape)

    anchor_labels_all = []
    for n in range(num_batch):
        anchor_labels = np.empty((len(anchors),), dtype=label_all[n].dtype)
        anchor_labels.fill(-1)
        anchor_labels[index_inside] = label_all[n]
        anchor_labels_all.append(anchor_labels)
    anchor_labels_all_merge = np.stack(anchor_labels_all, 0)
    # print(sum(anchor_labels_all[0]==1), anchor_labels_all[0][0:11])
    # print(anchor_labels_all_merge.shape)
    # print(sum(anchor_labels_all_merge[0]==1))

    anchor_locations_all = []
    for n in range(num_batch):
        anchor_locations = np.empty((len(anchors), anchors.shape[1]), dtype=anchor_locs_all[n].dtype)
        anchor_locations.fill(0)
        anchor_locations[index_inside, :] = anchor_locs_all[n]
        anchor_locations_all.append(anchor_locations)
    # print(anchor_locations_all[0].shape)
    # print(type(anchor_locations_all[0]))
    anchor_locations_all_merge = np.stack(anchor_locations_all, 0)
    # print(anchor_locations_all_merge[0][0])
    # print(anchor_locations_all[0][1500])

    return anchor_locations_all_merge, anchor_labels_all_merge, anchors