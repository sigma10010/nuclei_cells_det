from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

import numpy as np
from PIL import Image
from skimage import color, io

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import os
import torch
from torchvision import transforms

import shutil
import random

class RingCellNormalRegionDetection(Dataset):
    """
    General Histo dataset:
    input: root path of images, list to indicate indexes of slide
    output: a PIL Image and label
    """
    CLASSES = (
        "__background__ ",
        "ring_cell",
        
    )

    def __init__(self, root, transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images, root_dir/slides/images.
            slides (list): list of indexing slide
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root
        self._transforms = transforms
        
        cls = RingCellNormalRegionDetection.CLASSES
        self.category2id = dict(zip(cls, range(len(cls))))
        self.id2category = dict(zip(range(len(cls)), cls))
        

    def __len__(self):
        return len(self.walk_root_dir()[0])

    def __getitem__(self, idx):
        image = self.pull_item(idx)
        boxlist = None
        if self._transforms:
            image, boxlist = self._transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx
    
    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        image = self.pull_item(idx)
        img_width, img_height = image.size
        return {"height": img_height, "width": img_width}
    def map_class_id_to_class_name(self, class_id):
        return RingCellNormalRegionDetection.CLASSES[class_id]
    
    def toImg(self, x):
        t=transforms.ToPILImage()
        img = t(x)
        return img
        
    
    def walk_root_dir(self):
        names=[]
        wholePathes=[]
        
        annoNames=[]
        annoWholePathes=[]
        
        for dirpath, subdirs, files in os.walk(self.root_dir):
            for x in files:
                if x.endswith(('.png','.jpg', '.jpeg')):
                    names.append(x)
                    wholePathes.append(os.path.join(dirpath, x))
                    
                    y=x.split('.')[0]+'.xml'
                    annoNames.append(y)
                    annoWholePathes.append(os.path.join(dirpath, y))
        return names, wholePathes, annoNames, annoWholePathes
    
    def paser_xml(self, file):
        '''
        absolute coordinate [xmin, ymin, xmax, ymax]
        '''
        with open(file) as f:
            tree = ET.parse(f)
            objects=tree.findall('object')
            bndboxes=[]
            labels=[]
            difficults=[]
            for ob in objects:
                l=list(ob.find('bndbox'))
                xmin=int(l[0].text)
                ymin=int(l[1].text)
                xmax=int(l[2].text)
                ymax=int(l[3].text)
                bndbox=[xmin, ymin, xmax, ymax] # interger
                bndboxes.append(bndbox)
                labels.append(self.category2id['ring_cell'])
                difficults.append(int(ob.find('difficult').text)==1)
            return bndboxes, labels, difficults # list of list boxes [[xmin, ymin, xmax, ymax], ...]
        
    def get_groundtruth(self, index):
        image, boxes, lbs, dfclts = self.pull_item(index)
        labels = torch.from_numpy(np.array(lbs))
        difficults = torch.tensor(dfclts)
        # create a BoxList from the boxes
        # image need to be a PIL Image
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
        boxlist.add_field("difficult", difficults)
        return boxlist
        
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            

        images = torch.stack(images, dim=0)

        return images, boxes  # tensor (N, 3, 300, 300), 3 lists of N tensors each
    
    def pull_item(self, idx):
        """
        Args:
            index (int): idx
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        # load the image as a PIL Image
        imgNames, imgWholePathes, annoNames, annoWholePathes = self.walk_root_dir()
        img_name = imgWholePathes[idx]
        image = Image.open(img_name).convert("RGB")

        return image
        
    def pull_image(self, idx):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            numpy img
        '''
        imgNames, imgWholePathes, annoNames, annoWholePathes = self.walk_root_dir()
        img_name = imgWholePathes[idx]
        img=io.imread(img_name)
        img = np.array(img)
        return img

    def pull_anno(self, idx):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [xmin, ymin, xmax, ymax] # interger
        '''
        imgNames, imgWholePathes, annoNames, annoWholePathes = self.walk_root_dir()
        anno_name = annoWholePathes[idx]
        boxes, _, _ = self.paser_xml(anno_name)
        return boxes, imgNames[idx]
    
    def statistic(self):
        return 0

class RingCellDetection(Dataset):
    """
    General Histo dataset:
    input: root path of images, list to indicate indexes of slide
    output: a PIL Image and label
    """
    CLASSES = (
        "__background__ ",
        "ring_cell",
        
    )

    def __init__(self, root, transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images, root_dir/slides/images.
            slides (list): list of indexing slide
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root
        self._transforms = transforms
        
        cls = RingCellDetection.CLASSES
        self.category2id = dict(zip(cls, range(len(cls))))
        self.id2category = dict(zip(range(len(cls)), cls))
        

    def __len__(self):
        return len(self.walk_root_dir()[0])

    def __getitem__(self, idx):
        image, boxes, lbs, _ = self.pull_item(idx)
        labels = torch.from_numpy(np.array(lbs))
        
        # create a BoxList from the boxes
        # image need to be a PIL Image
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self._transforms:
            image, boxlist = self._transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx
    
    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        image, _, _, _ = self.pull_item(idx)
        img_width, img_height = image.size
        return {"height": img_height, "width": img_width}
    def map_class_id_to_class_name(self, class_id):
        return RingCellDetection.CLASSES[class_id]
    
    def toImg(self, x):
        t=transforms.ToPILImage()
        img = t(x)
        return img
        
    
    def walk_root_dir(self):
        names=[]
        wholePathes=[]
        
        annoNames=[]
        annoWholePathes=[]
        
        for dirpath, subdirs, files in os.walk(self.root_dir):
            for x in files:
                if x.endswith(('.png','.jpg', '.jpeg')):
                    names.append(x)
                    wholePathes.append(os.path.join(dirpath, x))
                    
                    y=x.split('.')[0]+'.xml'
                    annoNames.append(y)
                    annoWholePathes.append(os.path.join(dirpath, y))
        return names, wholePathes, annoNames, annoWholePathes
    
    def paser_xml(self, file):
        '''
        absolute coordinate [xmin, ymin, xmax, ymax]
        '''
        with open(file) as f:
            tree = ET.parse(f)
            objects=tree.findall('object')
            bndboxes=[]
            labels=[]
            difficults=[]
            for ob in objects:
                l=list(ob.find('bndbox'))
                xmin=int(l[0].text)
                ymin=int(l[1].text)
                xmax=int(l[2].text)
                ymax=int(l[3].text)
                bndbox=[xmin, ymin, xmax, ymax] # interger
                bndboxes.append(bndbox)
                labels.append(self.category2id['ring_cell'])
                # no use dificult
                difficults.append(int(ob.find('difficult').text)==1)
#                 difficults.append(int('0')==1)
            return bndboxes, labels, difficults # list of list boxes [[xmin, ymin, xmax, ymax], ...]
        
    def get_groundtruth(self, index):
        image, boxes, lbs, dfclts = self.pull_item(index)
        labels = torch.from_numpy(np.array(lbs))
        difficults = torch.tensor(dfclts)
        # create a BoxList from the boxes
        # image need to be a PIL Image
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
        boxlist.add_field("difficult", difficults)
        return boxlist
        
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            

        images = torch.stack(images, dim=0)

        return images, boxes  # tensor (N, 3, 300, 300), 3 lists of N tensors each
    
    def pull_item(self, idx):
        """
        Args:
            index (int): idx
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        # load the image as a PIL Image
        imgNames, imgWholePathes, annoNames, annoWholePathes = self.walk_root_dir()
        img_name = imgWholePathes[idx]
        image = Image.open(img_name).convert("RGB")
        # load the bounding boxes as a list of list of boxes
        anno_name = annoWholePathes[idx]
        boxes, labels, difficults = self.paser_xml(anno_name)

        return image, boxes, labels, difficults # [xmin, ymin, xmax, ymax] numpy image or torch depends on transform
        
    def pull_image(self, idx):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            numpy img
        '''
        imgNames, imgWholePathes, annoNames, annoWholePathes = self.walk_root_dir()
        img_name = imgWholePathes[idx]
        img=io.imread(img_name)
        img = np.array(img)
        return img

    def pull_anno(self, idx):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [xmin, ymin, xmax, ymax] # interger
        '''
        imgNames, imgWholePathes, annoNames, annoWholePathes = self.walk_root_dir()
        anno_name = annoWholePathes[idx]
        boxes, _, _ = self.paser_xml(anno_name)
        return boxes, imgNames[idx]
    
    def statistic(self):
        '''
        the number of cells
        '''
        n=0
        for i in range(self.__len__()):
            image, boxlist, idx = self.__getitem__(i)
            n+=len(boxlist)
        return n
    
class MoNuSegDetection(Dataset):
    """
    General Histo dataset:
    input: root path of images, list to indicate indexes of slide
    output: a PIL Image and label
    """
    CLASSES = (
        "__background__ ",
        "nucleus",
        
    )

    def __init__(self, img_root, anno_root, transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images, root_dir/slides/images.
            slides (list): list of indexing slide
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_root = img_root
        self.anno_root = anno_root
        self._transforms = transforms
        
        cls = MoNuSegDetection.CLASSES
        self.category2id = dict(zip(cls, range(len(cls))))
        self.id2category = dict(zip(range(len(cls)), cls))
        

    def __len__(self):
        return len(self.walk_root_dir()[0])

    def __getitem__(self, idx):
        image, anno = self.pull_item(idx)
        labels = torch.from_numpy(np.array(anno['labels']))
        
        # create a BoxList from the boxes
        # image need to be a PIL Image
        boxlist = BoxList(anno['bndboxes'], image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
        
        # add masks
        if anno and "polygons" in anno:
            masks = SegmentationMask(anno['polygons'], image.size, mode='poly')
            boxlist.add_field("masks", masks)

        if self._transforms:
            image, boxlist = self._transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx
    
    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        image, _ = self.pull_item(idx)
        img_width, img_height = image.size
        return {"height": img_height, "width": img_width}
    def map_class_id_to_class_name(self, class_id):
        return MoNuSegDetection.CLASSES[class_id]
    
    def toImg(self, x):
        t=transforms.ToPILImage()
        img = t(x)
        return img
        
    
    def walk_root_dir(self):
        names=[]
        wholePathes=[]
        annoNames=[]
        annoWholePathes=[]
        for dirpath, subdirs, files in os.walk(self.img_root):
            for x in files:
                if x.endswith(('.png','.jpg', '.jpeg', '.tif')):
                    names.append(x)
                    wholePathes.append(os.path.join(dirpath, x))
                    y = x.split('.')[0]+'.xml'
#                     y = x.split('.')[0]+'_det.xml'
                    annoNames.append(y)
                    annoWholePathes.append(os.path.join(dirpath, y))
#                     annoWholePathes.append(os.path.join(self.anno_root, y))
        return names, wholePathes, annoNames, annoWholePathes
    
    def paser_xml(self, file):
        '''
        absolute coordinate [xmin, ymin, xmax, ymax]
        '''
        with open(file) as f:
            tree = ET.parse(f)
            objects=tree.findall('object')
            bndboxes=[]
            labels=[]
            difficults=[]
            for ob in objects:
                l=list(ob.find('bndbox'))
                xmin=int(l[0].text)
                ymin=int(l[1].text)
                xmax=int(l[2].text)
                ymax=int(l[3].text)
                bndbox=[xmin, ymin, xmax, ymax] # interger
                bndboxes.append(bndbox)
                labels.append(self.category2id['nucleus'])
                # no use difficult
                difficults.append(int('0')==1)
#                 difficults.append(int(ob.find('difficult').text)==1)
            return {'bndboxes': bndboxes, 'labels': labels, 'difficults': difficults}

# paser segmentation xml
#             tree = ET.parse(f)
#             root = tree.getroot()
#             polygons = []
#             bndboxes=[]
#             labels=[]
#             difficults=[]
#             for region in root.iter('Region'):
#                 x=[]
#                 y=[]
#                 polygon=[] # [x1, y1, x2, y2, ..., xn, yn]
#                 for vertex in region.iter('Vertex'):
#                     x.append(float(vertex.attrib['X']))
#                     y.append(float(vertex.attrib['Y']))
#                     polygon.append(float(vertex.attrib['X']))
#                     polygon.append(float(vertex.attrib['Y']))
#                 polygons.append([polygon]) # [[[x1, y1, x2, y2, ..., xn, yn]], ...]
#                 x=np.array(x)
#                 y=np.array(y)
#                 xmin = x.min()
#                 xmax = x.max()
#                 ymin = y.min()
#                 ymax = y.max()
#                 bndbox=[xmin, ymin, xmax, ymax]
#                 bndboxes.append(bndbox)
#                 labels.append(self.category2id['nucleus'])
#                 difficults.append(int('0')==1)
#             return {'polygons': polygons, 'bndboxes': bndboxes, 'labels': labels, 'difficults': difficults} # list of list boxes [[xmin, ymin, xmax, ymax], ...]
        
    def get_groundtruth(self, index):
        image, anno = self.pull_item(index)
        labels = torch.from_numpy(np.array(anno['labels']))
        difficults = torch.tensor(anno['difficults'])
        # create a BoxList from the boxes
        # image need to be a PIL Image
        boxlist = BoxList(anno['bndboxes'], image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
        boxlist.add_field("difficult", difficults)
        return boxlist
        
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            

        images = torch.stack(images, dim=0)

        return images, boxes  # tensor (N, 3, 300, 300), 3 lists of N tensors each
    
    def pull_item(self, idx):
        """
        Args:
            index (int): idx
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        # load the image as a PIL Image
        imgNames, imgWholePathes, annoNames, annoWholePathes = self.walk_root_dir()
        img_name = imgWholePathes[idx]
        image = Image.open(img_name).convert("RGB")
        # load the bounding boxes as a list of list of boxes
        anno_name = annoWholePathes[idx]
        anno = self.paser_xml(anno_name)

        return image, anno # numpy image or torch depends on transform
        
    def pull_image(self, idx):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            numpy img
        '''
        imgNames, imgWholePathes, annoNames, annoWholePathes = self.walk_root_dir()
        img_name = imgWholePathes[idx]
        img=io.imread(img_name)
        img = np.array(img)
        return img

    def pull_anno(self, idx):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [xmin, ymin, xmax, ymax] # interger
        '''
        imgNames, imgWholePathes, annoNames, annoWholePathes = self.walk_root_dir()
        anno_name = annoWholePathes[idx]
        anno = self.paser_xml(anno_name)
        return anno['bndboxes'], imgNames[idx]
    
    def statistic(self):
        num_cell = 0
        for i in range(self.__len__()):
            num_cell += len(self.get_groundtruth(i))
        return num_cell

class divide_data():
    '''
    divide data by slide level
    '''
    def __init__(self, root_dir, rate=[1,1,1]):
        self.rate = rate  # train:val:test
        self.root = root_dir
        self.root_train = self.root+'train/'
        self.root_test = self.root+'test/'
        self.root_val = self.root+'validation/'
        self.root_all = self.root+'all/'
        if not os.path.exists(self.root_train):
            os.mkdir(self.root_train)
        if not os.path.exists(self.root_test):
            os.mkdir(self.root_test)
        if not os.path.exists(self.root_val):
            os.mkdir(self.root_val)

    def reset(self):
        for f in os.listdir(self.root_train):
            shutil.move(self.root_train+f,self.root_all)
        for f in os.listdir(self.root_test):
            shutil.move(self.root_test+f,self.root_all)
        for f in os.listdir(self.root_val):
            shutil.move(self.root_val+f,self.root_all)
        return 'sucess'

    def divide(self):   
        index=list(range(len(os.listdir(self.root_all))))
        random.shuffle(index)
        dirs=[]
        for _, f in enumerate(os.listdir(self.root_all)):
            dirs.append(f)
        for i in range(len(dirs)):
            if i<len(dirs)*(self.rate[0]/(np.array(self.rate).sum())):
                shutil.move(self.root_all+dirs[index[i]],self.root_train)
            elif len(dirs)*(self.rate[0]/np.array(self.rate).sum())<=i<len(dirs)-len(dirs)*(self.rate[2]/(np.array(self.rate).sum())):
                shutil.move(self.root_all+dirs[index[i]],self.root_val)
            else:
                shutil.move(self.root_all+dirs[index[i]],self.root_test)
        return 'sucess'


class PascalVOCDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "ring_cell",
        # "aeroplane",
        # "bicycle",
        # "bird",
        # "boat",
        # "bottle",
        # "bus",
        # "car",
        # "cat",
        # "chair",
        # "cow",
        # "diningtable",
        # "dog",
        # "horse",
        # "motorbike",
        # "person",
        # "pottedplant",
        # "sheep",
        # "sofa",
        # "train",
        # "tvmonitor",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return PascalVOCDataset.CLASSES[class_id]
