import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
from pathlib import Path
from tqdm import tqdm
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)
class LoadImgsAnnots(Dataset):
    def __init__(self, path, root_dir, set='train2017', transform=None, augment=True):
        
        path = str(Path(path))  # os-agnostic
        assert os.path.isfile(path), 'File not found %s. See %s' % (path, help_url)
        with open(path, 'r') as f:
            self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
                              if os.path.splitext(x)[-1].lower() in img_formats]

        n = len(self.img_files)
        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform
        self.cache_labels = True
        # self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        # self.image_ids = self.coco.getImgIds()
        # self.augment = augment
        # self.hyp = hyp
        # self.mosaic = self.augment 
        
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]
        # self.labels = [self.load_annotations(i) for i in self.image_ids] ## self.image_ids[image_index]
        self.labels = [None] * n
        if self.cache_labels:  # cache labels for faster training
            # self.labels = [] # annotations = np.zeros((0, 5))
            extract_bounding_boxes = False
            create_datasubset = False
            pbar = tqdm(self.label_files, desc='Caching labels')
            nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
            for i, file in enumerate(pbar):
                try:
                    with open(file, 'r') as f:
                        list = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    nm += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
                    continue

                if list.shape[0]:
                    assert list.shape[1] == 5, '> 5 label columns: %s' % file
                    assert (list >= 0).all(), 'negative labels: %s' % file
                    assert (list[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                    if np.unique(list, axis=0).shape[0] < list.shape[0]:  # duplicate rows
                        nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                    # if single_cls:
                    #     list[:, 0] = 0  # force dataset into single-class mode
                    
                    annotations = np.zeros((0, 5))
                    for annot in list: # lf: list in files
                        annotation = np.zeros((1, 5))
                        annotation[0, 0] = int(annot[0])
                        annotation[0, 1:] = annot[1:]
                        annotations = np.append(annotations, annotation, axis = 0)
                    self.labels[i] = annotations
                    nf += 1  # file found
                else:
                    ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                pbar.desc = 'Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                    nf, nm, ne, nd, n)
            assert nf > 0, 'No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        if not isSmallObjection(self, idx):
            img, annot = load_mosaic(self, idx)
        else:
            img, h, w = load_image(self, idx)
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.

            annotations = self.labels[idx]
            annot = np.zeros((0, 5))
            if annotations.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = annotations.copy()
                labels[:, 1] = w * (annotations[:, 1] - annotations[:, 3] / 2) 
                labels[:, 2] = h * (annotations[:, 2] - annotations[:, 4] / 2)
                labels[:, 3] = w * (annotations[:, 1] + annotations[:, 3] / 2) 
                labels[:, 4] = h * (annotations[:, 2] + annotations[:, 4] / 2)
            annot = np.append(annot, labels, axis = 0)
            del labels
        
        
        # img = img.numpy()

        # img = img.astype(np.float32) * 255.
        # for i in range(len(annot)):
        #     cv2.rectangle(img, (int(annot[i, 1]), int(annot[i, 2])), (int(annot[i, 3]), int(annot[i, 4])), (0, 0, 0), thickness=2)
        # cv2.imwrite('2.jpg', img)

        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        img = sample['img']
        img = img.numpy()
        img = img.astype(np.float32) * 255.
        annot = sample['annot']
        for i in range(len(annot)):
            cv2.rectangle(img, (int(annot[i, 1]), int(annot[i, 2])), (int(annot[i, 3]), int(annot[i, 4])), (255, 255, 255), thickness=2)
        img = cv2.imwrite('get2.jpg', img)

        return sample


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded}

def load_image(self, index):
    img_path = self.img_files[index]
    img = cv2.imread(img_path)  # BGR
    assert img is not None, 'Image Not Found ' + img_path
    h0, w0 = img.shape[:2]  # orig hw
    return img, h0, w0
    

def load_mosaic(self, index):

    labels4 = []

    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(5)]  # 3 additional image indices
    annotations4 = np.zeros((0, 5))
    for i, index in enumerate(indices):
        # Load image
        img, h, w = load_image(self, index)

        xc = w
        yc = 1.5 * h
        if i == 0:
            img6 = np.full((h * 3, w * 2, img.shape[2]), 114, dtype=np.uint8)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - 1.5 * h, 0), xc, yc - 0.5 * h
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - 1.5 * h, 0), min(xc + w, w * 2), yc - 0.5 * h
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc - 0.5 * h, xc, min(h * 3, yc + 0.5 * h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:
            x1a, y1a, x2a, y2a = xc, yc - 0.5 * h, min(xc + w, w * 2), min(h * 3, yc + 0.5 * h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        elif i == 4:
            x1a, y1a, x2a, y2a = max(xc - w, 0), min(yc + 0.5 * h, h * 3), xc, min(yc + 1.5 * h, h * 3)
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), min(y2a - y1a, h)
        elif i == 5:
            x1a, y1a, x2a, y2a = xc, min(yc + 0.5 * h, h * 3), min(xc + w, w * 2), min(yc + 1.5 * h, h * 3)
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), min(y2a - y1a, h)
        # print(y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b)
        img6[int(y1a):int(y2a), int(x1a):int(x2a)] = img[int(y1b):int(y2b), int(x1b):int(x2b)]  # img6[ymin:ymax, xmin:xmax]
        augment_hsv(img6, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
        img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2RGB)

        padw = x1a - x1b
        padh = y1a - y1b
        cv2.imwrite('mosaic.jpg', img6)
        
        annotations = self.labels[index]
        if annotations.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = annotations.copy()
            labels[:, 1] = w * (annotations[:, 1] - annotations[:, 3] / 2) + padw
            labels[:, 2] = h * (annotations[:, 2] - annotations[:, 4] / 2) + padh
            labels[:, 3] = w * (annotations[:, 1] + annotations[:, 3] / 2) + padw
            labels[:, 4] = h * (annotations[:, 2] + annotations[:, 4] / 2) + padh
        else:
            labels = np.zeros((0, 5), dtype=np.float32)
        
        annotations4 = np.append(annotations4, labels, axis = 0)
        del labels
    
    return img6.astype(np.float32) / 255., annotations4

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        start_h = np.random.randint(0, self.img_size - image.shape[0] + 1) # int(self.img_size / 2 - image.shape[0] / 2)
        end_h = start_h + image.shape[0]
        new_image[start_h:end_h, 0:resized_width] = image
        annots[:, 1:] *= scale
        annots[:, 2] += start_h 
        annots[:, 4] += start_h 
        # calculate areas threshold
        for annot in annots:
            area = (annot[4] - annot[2]) * (annot[3] - annot[1])
            if area < 270:
                annot[0] = -1
                for i in range(1, 5):
                    annot[i] = 0
        
        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots)}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 1].copy()
            x2 = annots[:, 3].copy()

            x_tmp = x1.copy()

            annots[:, 1] = cols - x2
            annots[:, 3] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):
    
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    x = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x).clip(None, 255).astype(np.uint8)
    np.clip(img_hsv[:, :, 0], None, 179, out=img_hsv[:, :, 0])  # inplace hue clip (0 - 179 deg)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def isSmallObjection(self, idx):
    annotations = self.labels[idx]
    if annotations.size > 0:
        # Normalized xywh to pixel xyxy format
        labels = np.array(annotations)
        area = np.array([1280 * labels[i, 2] * 720 * labels[i, 3] for i in range(len(labels))])
        if(len(area[area < 1500]) / len(labels)) > 0.3:
            return True