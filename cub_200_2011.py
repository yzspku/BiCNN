"""
Class of Dataset

call  get_train_validation_data_loader(resize_shape, batch_size, random_seed,
                                     augment=False, validation_size=0.3,
                                     object_boxes_dict=None,
                                     shuffle=True, show_sample=False)
      return  (train_loader, valid_loader)
call  get_test_data_loader(resize_shape, batch_size, object_boxes_dict=None, shuffle=True) 
      return (test_loader)

"""


import os

import torch
import numpy as np

import utils

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

use_less_data = False  # this flag is just for debugging multiple-process task
less_data_count_train = 500
less_data_count_test = 400

class BirdsDataset(Dataset):

  dataset_url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
  tar_file_name = 'CUB_200_2011.tgz'
  root_dir = 'CUB_200_2011'
  img_dir = 'images'
  data_split_file_name = 'train_test_split.txt'
  image_path_file_name = 'images.txt'
  image_label_file_name = 'image_class_labels.txt'
  classes_file_name = 'classes.txt'

  def __init__(self, object_boxes_dict=None, train=True, transform=None):
    self.train = train
    self.transform = transform
    self.download()  # download before loading

    train_indexes, test_indexes = self._get_train_test_indexes()
    img_label_dict = self._get_labels_of_images()
    img_path_dict = self._get_path_of_images()
    count = 0
    if train:
      self.image_indexes = []
      self.train_data = []
      self.train_labels = []
      for i in train_indexes:
        if use_less_data and count == less_data_count_train:
          break
        self.image_indexes.append(i)
        img_path = os.path.join(self.root_dir, self.img_dir, img_path_dict[i])
        img = self.__get_image_data(i, img_path, object_boxes_dict)
        self.train_data.append(img)
        self.train_labels.append(img_label_dict[i])
        count += 1
    else:
      self.image_indexes = []
      self.test_data = []
      self.test_labels = []
      for i in test_indexes:
        if use_less_data and count == less_data_count_test:
          break
        self.image_indexes.append(i)
        img_path = os.path.join(self.root_dir, self.img_dir, img_path_dict[i])
        img = self.__get_image_data(i, img_path, object_boxes_dict)
        self.test_data.append(img)
        self.test_labels.append(img_label_dict[i])
        count += 1

  def __getitem__(self, index):
    if self.train:
      data = self.train_data[index]
      label = self.train_labels[index]
    else:
      data = self.test_data[index]
      label = self.test_labels[index]
    if self.transform is not None:
      data = self.transform(data)
    return self.image_indexes[index], data, label

  def __len__(self):
    if self.train:
      return len(self.train_data)
    else:
      return len(self.test_data)

  def download(self):
    if self.__check_exist():
      return
    utils.download_file(self.dataset_url, self.tar_file_name)
    utils.extract_tgz(self.tar_file_name)

  def __check_exist(self):
    return os.path.exists(self.root_dir + '/images')

  def __get_image_data(self, img_idx, fpath, object_boxes_dict):
    img = Image.open(fpath)
    img = img.convert('RGB')  # it seems some of original images are png(???), which is ridiculous
    if object_boxes_dict is not None:
      box = object_boxes_dict[img_idx]  # (x, y, w, h)
      box = (box[0], box[1], box[0] + box[2], box[1] + box[3])  # (left, upper, right, lower)
      img = img.crop(box)
    return img

  def _get_train_test_indexes(self):
    fpath = os.path.join(self.root_dir, self.data_split_file_name)
    train_indexes = []
    test_indexes = []
    with open(fpath, 'r') as file:
      for line in file:
        tmp = line.split(' ')
        flag = tmp[1][0]
        if flag == '1':
          train_indexes.append(int(tmp[0]))
        else:
          test_indexes.append(int(tmp[0]))
    return train_indexes, test_indexes

  def _get_path_of_images(self):
    fpath = os.path.join(self.root_dir, self.image_path_file_name)
    img_path_dict = {}
    with open(fpath, 'r') as file:
      for line in file:
        tmp = line.split(' ')
        img_path_dict[int(tmp[0])] = tmp[1].strip('\n')
    return img_path_dict

  def _get_labels_of_images(self):
    fpath = os.path.join(self.root_dir, self.image_label_file_name)
    img_label_dict = {}
    with open(fpath, 'r') as file:
      for line in file:
        tmp = line.split(' ')
        img_label_dict[int(tmp[0])] = int(tmp[1]) - 1  # starts from 0, otherwise pytorch will throw an exception when training...
    return img_label_dict

  def get_classes_names(self):
    fpath = os.path.join(self.root_dir, self.classes_file_name)
    classes_names = []
    with open(fpath, 'r') as file:
      for line in file:
        tmp = line.split(' ')
        classes_names.append(tmp[1].strip('\n'))
    return classes_names


def get_train_validation_data_loader(resize_size, batch_size, random_seed,
                                     augment=False, validation_size=0.3,
                                     object_boxes_dict=None,
                                     shuffle=True, show_sample=False):
  normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
  )
  if augment:
    transforms_random_apply = transforms.RandomApply([
      transforms.RandomChoice([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomRotation(60)
      ]),
    ], p=0.4)
    if isinstance(resize_size, int):
      # shorter edges should be scaled to this size and original ratio will be kept
      # as a result, we should also do a random crop
      train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms_random_apply,
        transforms.RandomCrop(resize_size),
        transforms.ToTensor(),
        normalize
      ])
    else:  # should be a tuple like (224, 224)
      train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms_random_apply,
        transforms.ToTensor(),
        normalize
      ])
  else:
    if isinstance(resize_size, int):
      train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomCrop(resize_size),
        transforms.ToTensor(),
        normalize
      ])
    else:
      train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        normalize
      ])

  if isinstance(resize_size, int):  # for validation, we should keep all information of an image
    resize_size = (resize_size, resize_size)
  valid_transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
    normalize
  ])

  train_dataset = BirdsDataset(
    train=True, transform=train_transform, object_boxes_dict=object_boxes_dict)
  valid_dataset = BirdsDataset(
    train=True, transform=valid_transform, object_boxes_dict=object_boxes_dict)

  num_train = len(train_dataset)
  indices = list(range(num_train))
  split = int(np.floor(validation_size * num_train))

  if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

  train_idx, valid_idx = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)

  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4
  )
  valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4
  )

  # visualize some images
  if show_sample:
    sample_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=9, shuffle=shuffle
    )
    data_iter = iter(sample_loader)
    images, labels = data_iter.next()
    X = images.numpy().transpose([0, 2, 3, 1])
    utils.plot_images(train_dataset.get_classes_names(), X, labels)

  return train_loader, valid_loader


def get_test_data_loader(resize_size, batch_size, object_boxes_dict=None, shuffle=True):
  normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
  )
  if isinstance(resize_size, int):
    resize_size = (resize_size, resize_size)
  transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
    normalize
  ])
  test_dataset = BirdsDataset(
    train=False, transform=transform, object_boxes_dict=object_boxes_dict)
  test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=shuffle
  )
  return test_loader
