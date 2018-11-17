import cub_200_2011 as dataset
import helper
import torch
import utils
import time

model_glb_path = ''
model_obj_path = ''

predict_weights = [0.2, 0.8]
logger = utils.get_logger('weights-' + str(predict_weights[0]) + '-' + str(predict_weights[1]))
logger.info('start loading dataset')
begin_time = time.time()
train_loader_glb, valid_loader_glb = dataset.get_train_validation_data_loader(
  resize_size=224,
  batch_size=32,
  random_seed=96,
  validation_size=0.1,
  object_boxes_dict=None,
  show_sample=False
)
test_loader_glb = dataset.get_test_data_loader(
  resize_size=224,
  batch_size=32,
  object_boxes_dict=None
)

bounding_boxes = utils.get_annotated_bounding_boxes()
train_loader_obj, valid_loader_obj = dataset.get_train_validation_data_loader(
  resize_size=224,
  batch_size=32,
  random_seed=96,
  validation_size=0.1,
  object_boxes_dict=bounding_boxes,
  show_sample=False
)
test_loader_obj = dataset.get_test_data_loader(
  resize_size=224,
  batch_size=32,
  object_boxes_dict=bounding_boxes
)
logger.info('loading dataset costs %.4fs' % (time.time() - begin_time))

logger.info('loading models')

begin_time = time.time()
model_glb_name = 'resnet152'
model_glb = helper.get_model_by_name(model_glb_name, pretrained=False)
helper.replace_model_fc(model_glb_name, model_glb)
model_glb.load_state_dict(torch.load(model_glb_path))

model_obj_name = 'densenet161'
model_obj = helper.get_model_by_name(model_obj_name, pretrained=False)
helper.replace_model_fc(model_obj_name, model_obj)
model_obj.load_state_dict(torch.load(model_obj_path))
logger.info('loading models costs %.4fs' % (time.time() - begin_time))

models = [model_glb, model_obj]
validation_loaders = [valid_loader_glb, valid_loader_obj]
test_loaders = [test_loader_glb, test_loader_obj]

helper.evaluate(
  logger=logger,
  models=models,
  data_loaders=validation_loaders,
  set_name='validation set',
  predict_weights=predict_weights
)
helper.evaluate(
  logger=logger,
  models=models,
  data_loaders=test_loaders,
  set_name='test set',
  predict_weights=predict_weights
)