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
        resnet_model = torchvision.models.resnet34(pretrained=False)
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
    def __init__(self, path):
        """Prepare the network, criterion, solver, and data.

        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._path = path
        # Network.
        self._net = torch.nn.DataParallel(BCNN()).cuda()
        self._net.module.freeze_layers()
        self._net.load_state_dict(torch.load(self._path))

        self._test_loader = dataset.get_test_data_loader(
            resize_size=448,
            batch_size=32,
            object_boxes_dict=None
        )

    def test(self):
        """Train the network."""
        print('Testing.')
        test_acc = 1.0 * self._accuracy(self._test_loader)
        print("Test acc: %.4f" % test_acc)

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
        return 100.0 * num_correct / num_total


def test():

    path_save='models/resnet_34_all.pth'
    manager = BCNNManager(path_save)
    manager.test()

if __name__ == '__main__':
    #dataset.use_less_data=True
    test()
