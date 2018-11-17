# -*- coding: utf-8 -*-

import os

import torch
import torchvision
import time
import cub_200_2011 as dataset

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class BCNN(torch.nn.Module):

  def __init__(self):
    """Declare all needed layers."""
    torch.nn.Module.__init__(self)
    resnet_model = torchvision.models.resnet34(pretrained=True)
    self.conv1 = resnet_model.conv1
    self.bn1 = resnet_model.bn1
    self.relu = resnet_model.relu
    self.maxpool = resnet_model.maxpool
    self.layer1 = resnet_model.layer1
    self.layer2 = resnet_model.layer2
    self.layer3 = resnet_model.layer3
    self.layer4 = resnet_model.layer4
    # Linear classifier.
    self.fc = torch.nn.Linear(512**2, 200)
    # Initialize the fc layers.
    torch.nn.init.kaiming_normal_(self.fc.weight.data)
    if self.fc.bias is not None:
      torch.nn.init.constant_(self.fc.bias.data, val=0)

  def forward(self, X):

    N = X.size()[0]
    assert X.size() == (N, 3, 448, 448)
    x = self.conv1(X)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    X = self.layer4(x)
    assert X.size() == (N, 512, 14, 14)
    X = X.view(N, 512, 14**2)
    X = torch.bmm(X, torch.transpose(X, 1, 2)) / (14**2)  # Bilinear
    assert X.size() == (N, 512, 512)
    X = X.view(N, 512**2)
    X = torch.sqrt(X + 1e-5)
    X = torch.nn.functional.normalize(X)
    X = self.fc(X)
    assert X.size() == (N, 200)
    return X
  def freeze_layers(self):
    # Freeze all previous layers.
    for param in self.conv1.parameters():
      param.requires_grad = False
    for param in self.bn1.parameters():
      param.requires_grad = False
    for param in self.layer1.parameters():
      param.requires_grad = False
    for param in self.layer2.parameters():
      param.requires_grad = False
    for param in self.layer3.parameters():
      param.requires_grad = False
    for param in self.layer4.parameters():
      param.requires_grad = False

class BCNNManager(object):
  """Manager class to train bilinear CNN.

  Attributes:
      _options: Hyperparameters.
      _path: Useful paths.
      _net: Bilinear CNN.
      _criterion: Cross-entropy loss.
      _solver: SGD with momentum.
      _scheduler: Reduce learning rate by a fator of 0.1 when plateau.
      _train_loader: Training data.
      _test_loader: Testing data.
  """
  def __init__(self, options, path, freeze=True, pre_model_path=None):
    """Prepare the network, criterion, solver, and data.

    Args:
        options, dict: Hyperparameters.
    """
    print('Prepare the network and data.')
    self._options = options
    self._path = path
    # Network.
    self._net = torch.nn.DataParallel(BCNN()).cuda()
    #print(self._net)
    if freeze is True:
      self._net.module.freeze_layers()
    if pre_model_path is not None:
      self._net.load_state_dict(torch.load(pre_model_path))
    # Criterion.
    self._criterion = torch.nn.CrossEntropyLoss().cuda()
    # Solver.
    if freeze is True:
      self._solver = torch.optim.SGD(
        self._net.module.fc.parameters(), lr=self._options['base_lr'],
        momentum=0.9, weight_decay=self._options['weight_decay'])
    else:
      self._solver = torch.optim.SGD(
        self._net.parameters(), lr=self._options['base_lr'],
        momentum=0.9, weight_decay=self._options['weight_decay'])
    self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      self._solver, mode='max', factor=0.1, patience=3, verbose=True,
      threshold=1e-4)
    self._train_loader, self._valid_loader = dataset.get_train_validation_data_loader(
      resize_size=448,
      batch_size=self._options['batch_size'],
      random_seed=96,
      validation_size=0,
      object_boxes_dict=None,
      show_sample=False,
      augment=True
    )
    self._test_loader = dataset.get_test_data_loader(
      resize_size=448,
      batch_size=32,
      object_boxes_dict=None
    )

  def train(self):
    """Train the network."""
    print('Training.')
    best_acc = 0.0
    best_epoch = None
    print('Epoch\tTrain loss\tTrain acc\tValid acc\tTest acc')
    for t in range(self._options['epochs']):
      epoch_loss = []
      num_correct = 0
      num_total = 0
      for i, (_, X, y) in enumerate(self._train_loader, 0):
        # Data.
        X = torch.autograd.Variable(X.cuda())
        y = torch.autograd.Variable(y.cuda(async=True))

        # Clear the existing gradients.
        self._solver.zero_grad()
        # Forward pass.
        score = self._net(X)
        loss = self._criterion(score, y)
        epoch_loss.append(loss.data.item())
        # Prediction.
        _, prediction = torch.max(score.data, 1)
        num_total += y.size(0)
        num_correct += torch.sum(prediction == y.data).float()
        # Backward pass.
        loss.backward()
        self._solver.step()
      train_acc = 100.0 * num_correct / num_total
      #valid_acc = 1.0 * self._accuracy(self._valid_loader)
      test_acc = 1.0 * self._accuracy(self._test_loader)
      self._scheduler.step(test_acc)
      if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = t + 1
        print('*', end='')
        # Save model onto disk.
        torch.save(self._net.state_dict(),self._path)
      print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
            (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
    print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

  def _accuracy(self, data_loader):
    """Compute the train/test accuracy.

    Args:
        data_loader: Train/Test DataLoader.

    Returns:
        Train/Test accuracy in percentage.
    """
    self._net.train(False)
    num_correct = 0
    num_total = 0
    for i, (_, X, y) in enumerate(data_loader, 0):
      # Data.
      X = torch.autograd.Variable(X.cuda())
      y = torch.autograd.Variable(y.cuda(async=True))

      # Prediction.
      score = self._net(X)
      _, prediction = torch.max(score.data, 1)
      num_total += y.size(0)
      num_correct += torch.sum(prediction == y.data).float()
    self._net.train(True)  # Set the model to training phase
    return 100.0 * num_correct / num_total

def fc():
  options = {
    'base_lr': 1.0,
    'batch_size': 64,
    'epochs': 50,
    'weight_decay': 1e-8,
  }
  pre_model_path = None
  path_save='models/resnet34_fc_448.pth'
  manager = BCNNManager(options, path_save, freeze=True, pre_model_path=pre_model_path)
  manager.train()

def all_layers():
  options = {
    'base_lr': 0.01,
    'batch_size': 32,
    'epochs': 30,
    'weight_decay': 1e-5,
  }
  pre_model_path = 'models/resnet34_fc_448.pth'
  path_save='models/resnet34_all_448.pth'
  manager = BCNNManager(options, path_save, freeze=False, pre_model_path=pre_model_path)
  #manager.train()

if __name__ == '__main__':
  #fc()
  #torch.cuda.empty_cache()
  all_layers()

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class ScaledDotProductAttention(nn.Module):
  ''' Scaled Dot-Product Attention '''

  def __init__(self, temperature, attn_dropout=0.1):
    super().__init__()
    self.temperature = temperature
    self.dropout = nn.Dropout(attn_dropout)
    self.softmax = nn.Softmax(dim=2)

  def forward(self, q, k, v, mask=None):

    attn = torch.bmm(q, k.transpose(1, 2))
    attn = attn / self.temperature

    if mask is not None:
      attn = attn.masked_fill(mask, -np.inf)

    attn = self.softmax(attn)
    attn = self.dropout(attn)
    output = torch.bmm(attn, v)

    return output, attn

class MultiHeadAttention(nn.Module):
  ''' Multi-Head Attention module '''

  def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
    super().__init__()

    self.n_head = n_head
    self.d_k = d_k
    self.d_v = d_v

    self.w_qs = nn.Linear(d_model, n_head * d_k)
    self.w_ks = nn.Linear(d_model, n_head * d_k)
    self.w_vs = nn.Linear(d_model, n_head * d_v)
    nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
    nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
    nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
    self.layer_norm = nn.LayerNorm(d_model)

    self.fc = nn.Linear(n_head * d_v, d_model)
    nn.init.xavier_normal_(self.fc.weight)

    self.dropout = nn.Dropout(dropout)


  def forward(self, q, k, v, mask=None):

    d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

    sz_b, len_q, _ = q.size() # batch, 79, 256
    sz_b, len_k, _ = k.size()
    sz_b, len_v, _ = v.size()

    residual = q

    q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # N, 79, 8, 64
    k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
    v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

    q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
    k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
    v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

    mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
    output, attn = self.attention(q, k, v, mask=mask)

    output = output.view(n_head, sz_b, len_q, d_v)
    output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

    output = self.dropout(self.fc(output))
    output = self.layer_norm(output + residual)

    return output, attn
