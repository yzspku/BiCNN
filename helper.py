'''
1.Valuate trained model(s) on dataset(s) and 
combine their results into one output prediction

def evaluate(log, 
  models, 
  data_loaders, 
  set_name, 
  predict_weights=None,
  use_gpu=cuda.is_available(), 
  cuda_device_idx=0):

  return acc


2.Train a model and evaluate it after training

train_and_evaluate(
  log = None,
  model_name = 'resnet152',
  pre_model = None,
  use_pretrained_params = True,
  fine_tune_all_layers = False,

  data_loaders=None,
  is_object_level=False,

  num_epochs = 4,
  learning_rate = 1e-3,
  weight_decay = 5e-4,
  train_batch_size = 32,
  eval_epoch_step = 4,

  use_gpu = cuda.is_available(),
  cuda_device_idx = 0,
  use_multiple_gpu = False,

  save_model = True
):

return model, train_acc, valid_acc, test_acc, model_path


'''

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from torchvision.models import *
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import os, time

import cub_200_2011 as dataset
import utils

def get_model_by_name(name, pretrained):
  if name == 'resnet18':    return resnet18(pretrained=pretrained)
  if name == 'resnet34':    return resnet34(pretrained=pretrained)
  if name == 'resnet50':    return resnet50(pretrained=pretrained)
  if name == 'resnet101':   return resnet101(pretrained=pretrained)
  if name == 'resnet152':   return resnet152(pretrained=pretrained)
  if name == 'vgg16':       return vgg16_bn(pretrained=pretrained)
  if name == 'vgg19':       return vgg19_bn(pretrained=pretrained)
  if name == 'inception':   return inception_v3(pretrained=pretrained)
  if name == 'densenet121': return densenet121(pretrained=pretrained)
  if name == 'densenet169': return densenet169(pretrained=pretrained)
  if name == 'densenet201': return densenet201(pretrained=pretrained)
  if name == 'densenet161': return densenet161(pretrained=pretrained)

#in  order to modify resnet.fc , used in 'replace_model_fc'
resnet_block_dict = {
  'resnet18': 1, 'resnet34': 1, 'resnet50': 1,
  'resnet101': 4, 'resnet152': 4,
}
def replace_model_fc(model_name, model):
  """ Replace fully connected layer of a neural network model in order to correct output class number

  :param model_name: model's name
  :param model: model itself, a pytorch's nn.Module object
  """
  if model_name.startswith('bilinear_densenet'):
    for param in model.conv.parameters():
      param.requires_grad = True
    for param in model.bn.parameters():
      param.requires_grad = True
    for param in model.fc.parameters():
      param.requires_grad = True
    return None

  # change the num_classes to 200
  if model_name.startswith('resnet'):
    model.fc = nn.Linear(512 * resnet_block_dict[model_name], 200)
  elif model_name.startswith('vgg'):
    pass  # todo find out how we can change num_classes to fine tune vgg
  elif model_name == 'inception':
    model.fc = nn.Linear(2048, 200)
  elif model_name.startswith('densenet'):
    model.classifier = nn.Linear(model.classifier.in_features, 200)
  elif model_name.startswith('bilinear_resnet'):
    if model_name.endswith('152'):
      model.conv2 = nn.Conv2d(2048, 512, 1)
      model.bn2 = nn.BatchNorm2d(512)
    model.fc = nn.Linear(512**2, 200)


def get_model_parameters(model_name, model, pretrained, fine_tune_all_layers):
  """ Get model's parameters to optimize

  :param model_name: model's name
  :param model: model itself
  :param pretrained: True if we should use pretrained model parameters
  :param fine_tune_all_layers: True if we should fine tune all layers of the model
  """
  if not pretrained or fine_tune_all_layers:
    return model.parameters()
  else:  # fine tune only fully connected layer
    if model_name.startswith('resnet') or model_name.startswith('inception'):
      return model.fc.parameters()
    elif model_name.startswith('densenet'):
      return model.classifier.parameters()
    elif model_name.startswith('bilinear_densenet'):
      return list(model.conv.parameters()) + list(model.bn.parameters()) +  list(model.fc.parameters())
    elif model_name.startswith('bilinear'):
      if model_name.endswith('34'):
        return model.fc.parameters()
      elif model_name.endswith('152'):
        return list(model.conv2.parameters()) + list(model.bn2.parameters()) + list(model.fc.parameters())
    else:  # vgg
      pass


def save_model_parameters(parameters, file_name_prefix):
  # parameters should come from model.state_dict()
  if not os.path.exists('models/'):
    os.makedirs('models/')
  fp = 'models/' + file_name_prefix + '_' + time.strftime("%m-%d-%H-%M", time.localtime()) + '.pth'
  torch.save(parameters, fp)
  return fp


def save_evaluation_result(prefix, epochs_arr, losses, epochs_step_arr, train_accuracies, valid_accuracies):
  if not os.path.exists("result"):
    os.mkdir("result")

  post_fix = time.strftime("%m-%d-%H-%M", time.localtime())

  plt.clf()  # clear existing figure content
  plt.plot(epochs_arr, losses)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.savefig("result/" + prefix + "_loss_" + post_fix + ".png")

  plt.clf()  # clear existing figure content
  plt.plot(epochs_step_arr, train_accuracies)
  plt.xlabel('epoch')
  plt.ylabel('train accuracy')
  plt.savefig("result/" + prefix + "_acc_train_" + post_fix + ".png")

  plt.clf()  # clear existing figure content
  plt.plot(epochs_step_arr, valid_accuracies)
  plt.xlabel('epoch')
  plt.ylabel('validation accuracy')
  plt.savefig("result/" + prefix + "_acc_valid_" + post_fix + ".png")


def predict(model_glb, img_pil, resize_shape=(224, 224), model_obj=None, obj_bounding_box=None, predict_weights=None, use_gpu=cuda.is_available(), cuda_device_idx=0):
  """ predict input's class

  :param model_glb: classification model for global level
  :param img_pil: image as PIL.Image
  :param resize_shape: resize shape
  :param model_obj: classification model for object level
  :param obj_bounding_box: as its name
  :param predict_weights: weights of different levels' prediction, [0] should be for global level
  :param use_gpu: as its name
  :param cuda_device_idx: as its name
  :return: the top5 [probabilities(???), indices] list
  """

  img = img_pil.resize(resize_shape)
  img_obj = None
  if obj_bounding_box is not None:
    img_obj = img_pil.crop(obj_bounding_box)
    img_obj = img_obj.resize(resize_shape)

  normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
  )
  transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
  ])

  img_tensor_glb = transform(img)
  # tmp = img_tensor_glb.numpy().transpose([1, 2, 0])
  # plt.imshow(tmp)
  # plt.show()

  img_tensor_glb = img_tensor_glb.unsqueeze(0)  # convert shape (3, 224, 224) to (1, 3, 224, 224)
  img_tensor_obj = None
  if model_obj is not None:
    img_tensor_obj = transform(img_obj)
    # tmp = img_tensor_obj.numpy().transpose([1, 2, 0])
    # plt.imshow(tmp)
    # plt.show()
    img_tensor_obj = img_tensor_obj.unsqueeze(0)  # convert shape (3, 224, 224) to (1, 3, 224, 224)

  if use_gpu:
    cuda_device = torch.device('cuda', cuda_device_idx)
    img_tensor_glb = img_tensor_glb.cuda(cuda_device)
    model_glb = model_glb.cuda(cuda_device_idx)
    if model_obj is not None:
      img_tensor_obj = img_tensor_obj.cuda(cuda_device)
      model_obj = model_obj.cuda(cuda_device_idx)

  predict_prob_arr_glb = model_glb(img_tensor_glb)  # probabilities
  predict_prob_arr = predict_prob_arr_glb
  if model_obj is not None:
    predict_prob_arr_obj = model_obj(img_tensor_obj)
    predict_prob_arr = predict_weights[0] * predict_prob_arr_glb + predict_weights[1] * predict_prob_arr_obj
  top5 = torch.topk(predict_prob_arr, 5)
  if use_gpu:
    top5 = [top5[0].cpu(), top5[1].cpu()]  # back to cpu so that we can detach them
  probs = top5[0].detach().numpy()  # 2-d nparray
  classes = top5[1].detach().numpy()
  probs = probs[0]
  classes = classes[0]
  list_prob = [probs[0], probs[1], probs[2], probs[3], probs[4], ]
  list_cls = [classes[0] + 1, classes[1] + 1, classes[2] + 1, classes[3] + 1, classes[4] + 1, ]
  return [list_prob, list_cls]


def evaluate(logger, models, data_loaders, set_name, predict_weights=None, use_gpu=cuda.is_available(), cuda_device_idx=0,
             use_multiple_gpu=False):
  """ Evaluate trained model(s) on dataset(s) and combine their results into one output prediction
  Note: there should be an one-to-one match between models and data_loaders.

  :param logger: the utils.LoggerS object to print logs
  :param models: a list of models
  :param data_loaders: a list of dataset loaders.
  :param set_name: dataset's name, can be 'train_set', 'validation set' or 'test set'

  :param predict_weights: a list of weights for each models' prediction result
    Example: for a specific input, models[0] gives an output as [0.7, 0.3] (probability of class 0 and class 1),
    and models[1] gives [0.4, 0.6]. If the predict_weights is [0.2, 0.8], then the final output should be
    [0.7*0.2 + 0.4*0.8, 0.3*0.2 + 0.6*0.8], i.e. [0.46, 0.54], so the prediction is class 1

  :param use_gpu: use GPU to run the model or not
  :param cuda_device_idx: an int value that indicates which cuda device that we want to use for inputs

  :return: prediction accuracy
  """
  for model in models:
    model.eval()  # evaluation mode

  if predict_weights is None:
    predict_weights = [1]
  logger.info('computing classification accuracy on ' + set_name)
  _begin_time = time.time()
  #acc = correct_num / sample_num
  correct_num = 0
  sample_num = 0
  has_multiple_gpu = cuda.device_count() > 1
  cuda_device = None
  if use_gpu:
    cuda_device = torch.device('cuda', cuda_device_idx)
  with torch.no_grad():
    labels_dict = {}
    predicts_dict = {}
    for i in range(len(models)):  # each model is only valid on corresponding data loader
      model = models[i]
      data_loader = data_loaders[i]
      for data in data_loader:
        image_indexes, images, labels = data
        if has_multiple_gpu and use_gpu:
          if use_multiple_gpu:
            images = torch.autograd.Variable(images.cuda())
            labels = torch.autograd.Variable(labels.cuda(async=True))
          else:
            images = images.cuda(cuda_device)
            labels = labels.cuda(cuda_device)  # shape is (batch_size, 1)

        batch_size = labels.size(0)
        predict = model(images)  # shape is (batch_size, 200)

        if i == 0:
          sample_num += labels.size(0)
          for j in range(batch_size):
            img_idx = image_indexes[j].item()
            labels_dict[img_idx] = labels[j].item()
            predicts_dict[img_idx] = predict_weights[i] * predict.data[j]
        else:
          for j in range(batch_size):
            img_idx = image_indexes[j].item()
            predict_data = predict_weights[i] * predict.data[j] + predicts_dict[img_idx]
            predicts_dict[img_idx] = predict_data

  for img_idx in predicts_dict:
    label = labels_dict[img_idx]
    predict = predicts_dict[img_idx]
    predict_cls = torch.argmax(predict)
    if predict_cls == label:
      correct_num += 1

  for model in models:
    model.train()  # back to train mode

  acc = 100.0 * correct_num / sample_num
  logger.info('accuracy: %.4f%%, cost time: %.4fs' % (acc, time.time() - _begin_time))
  return acc


# ----------------------- This is a very important method ----------------------

def train_and_evaluate(
        logger = None,

        model_name = 'resnet152',
        pre_model = None,
        use_pretrained_params = True,
        fine_tune_all_layers = False,

        data_loaders=None,
        is_object_level=False,

        num_epochs = 4,
        learning_rate = 1e-3,
        use_scheduler = False,
        weight_decay = 5e-4,
        train_batch_size = 32,
        eval_epoch_step = 4,

        use_gpu = cuda.is_available(),
        cuda_device_idx = 0,
        use_multiple_gpu = False,

        save_model = True
):
  """ Train a model and evaluate it after training

  :param logger: the utils.LoggerS object to print logs onto file and console
  :param model_name: model's name, used to create the model and help provide more detailed log
  :param pre_model: if this is not None, we will train and evaluate on it instead of creating a new model
  :param use_pretrained_params: True if we initialize the model with pretrained parameters
  :param fine_tune_all_layers: True if we want to fine tune all layers of the model

  :param data_loaders: a list of data loaders for train, validation and test set. The order must be correct
  :param is_object_level: as its name

  :param num_epochs: the number of training iterations on whole train set
  :param learning_rate: as its name
  :param weight_decay: as its name
  :param train_batch_size: batch size of train set
  :param eval_epoch_step: evaluation step

  :param use_gpu: use GPU to train/evaluate or not
  :param cuda_device_idx: an int value that indicates which cuda device that we want to use for inputs and model
  :param use_multiple_gpu: use multiple GPU to train/evaluate or not; todo currently this flag is useless

  :param save_model: True if we want to save the model that has best validation accuracy when training

  :return: trained model, accuracies on train, validation and test set,
            and stored model path if :param save_model is set to True
  """
  # obj -- object
  # glb -- global
  # prtrn -- pretrain
  # ep -- epoch
  # bt -- batch_size
  if is_object_level:
    res_file_name_prefix = 'obj'
  else:
    res_file_name_prefix = 'glb'
  res_file_name_prefix += '_' + model_name
  if use_pretrained_params:
    res_file_name_prefix += '_prtrn'
    if fine_tune_all_layers:
      res_file_name_prefix += 'All'
  res_file_name_prefix += '_ep' + str(num_epochs) + '_bt' + str(train_batch_size) + '_' + str(learning_rate)
  if logger is None:
    logger = utils.get_logger(res_file_name_prefix)

  # get train/valid/test_loader
  if data_loaders is None:
    logger.info('start loading dataset')
    begin_time = time.time()
    train_loader, valid_loader = dataset.get_train_validation_data_loader(
      resize_size=224,
      batch_size=train_batch_size,
      random_seed=96,
      validation_size=0.2,
      object_boxes_dict=None,
      show_sample=False
    )
    test_loader = dataset.get_test_data_loader(
      resize_size=224,
      batch_size=32,
      object_boxes_dict=None
    )
    logger.info('loading dataset costs ' + str(time.time() - begin_time))
  else:
    train_loader = data_loaders[0]
    valid_loader = data_loaders[1]
    test_loader  = data_loaders[2]

  # Create nn model
  if pre_model is not None:
    model = pre_model
    # pre_model should have been trained
    if not fine_tune_all_layers:
      for param in model.parameters():
        param.requires_grad = False
      replace_model_fc(model_name, model)
  else:
    model = get_model_by_name(model_name, use_pretrained_params)
    if use_pretrained_params and not fine_tune_all_layers:
      # only fine tune fully connected layer, which means we should not upgrade network layers except for last one
      for param in model.parameters():
        param.requires_grad = False
    replace_model_fc(model_name, model)

  has_multiple_gpu = cuda.device_count() > 1

  cuda_device = None  # declare this just in order to remove IDE warnings ...
  if use_gpu:
    if has_multiple_gpu and use_multiple_gpu: model = nn.DataParallel(model).cuda()
    else :
      model = model.cuda(cuda_device_idx)
      cuda_device = torch.device('cuda', cuda_device_idx)


  criterion = nn.CrossEntropyLoss().cuda()
  if has_multiple_gpu and use_multiple_gpu: _model=model.module
  else:_model=model
  optimizer = optim.SGD(
    get_model_parameters(model_name, _model, use_pretrained_params, fine_tune_all_layers),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=weight_decay
  )
  # Reduce learning rate when a metric has stopped improving.
  if use_scheduler is True:
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
          optimizer, mode='min', factor=0.1, patience=3, verbose=True, threshold=1e-4
      )


  logger.info('start training')
  train_cost_time = 0.0
  epochs_arr = []
  losses_arr = []
  epochs_step_arr = []
  train_acc_arr = []
  valid_acc_arr = []
  best_valid_acc = 0.0
  best_valid_acc_model_params = None
  for epoch in range(num_epochs):
    running_loss = 0.0
    batch_num = 0
    for i, (_, inputs, labels) in enumerate(train_loader, 0):
      begin_time = time.time()
      # get the inputs

      if use_gpu:
        if has_multiple_gpu and use_multiple_gpu:
          inputs = torch.autograd.Variable(inputs.cuda())
          labels = torch.autograd.Variable(labels.cuda(async=True))
        else:
          inputs = inputs.cuda(cuda_device)
          labels = labels.cuda(cuda_device)


      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      # print statistics
      logger.info('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, loss.item()))
      cost_time_i = time.time() - begin_time
      train_cost_time += cost_time_i
      logger.info('cost time: %.4fs' % cost_time_i)
      batch_num = i
    if use_scheduler is True:
        scheduler.step(running_loss)
    epochs_arr.append(epoch + 1)
    losses_arr.append(running_loss / batch_num)
    if epoch == 0 or (epoch + 1) % eval_epoch_step == 0:  # compute classification accuracy on train and validation set
      epochs_step_arr.append(epoch + 1)
      logger.info('')
      train_acc = evaluate(logger=logger, models=[model], data_loaders=[train_loader],
                           set_name='train set', cuda_device_idx=cuda_device_idx, use_multiple_gpu=use_multiple_gpu)
      train_acc_arr.append(train_acc)
      valid_acc = evaluate(logger=logger, models=[model], data_loaders=[valid_loader],
                           set_name='validation set', cuda_device_idx=cuda_device_idx, use_multiple_gpu=use_multiple_gpu)
      valid_acc_arr.append(valid_acc)
      if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_valid_acc_model_params = model.state_dict()
      logger.info('')

  logger.info('Finished Training, cost time: %.4fs' % train_cost_time)
  logger.info('')

  test_acc = evaluate(logger=logger, models=[model], data_loaders=[test_loader],
                      set_name='test set', cuda_device_idx=cuda_device_idx, use_multiple_gpu=use_multiple_gpu)
  logger.info('')

  save_evaluation_result(res_file_name_prefix, epochs_arr, losses_arr, epochs_step_arr, train_acc_arr, valid_acc_arr)

  saved_model_path = None
  if save_model:
    logger.info('')
    logger.info('saving model parameters')
    if is_object_level:
      model_file_name_prefix = 'obj_'
    else:
      model_file_name_prefix = 'glb_'
    model_file_name_prefix += model_name + ('_acc%.4f' % best_valid_acc)
    saved_model_path = save_model_parameters(best_valid_acc_model_params, model_file_name_prefix)
    logger.info('parameters have been saved successfully to ' + saved_model_path)
    logger.info('')

  return model, train_acc_arr[len(train_acc_arr) - 1], valid_acc_arr[len(valid_acc_arr) - 1], test_acc, saved_model_path

