import cub_200_2011 as dataset
import helper, utils
import model_global as glb, model_object as obj
import time
import torch
import torch.multiprocessing as mp

train_batch_size = 32
test_batch_size = 32
random_seed = 96
validation_size = 0.1
predict_weights = [0.2, 0.8]

use_multiple_gpu = False  # run global model on 1 gpu and object-level model on another
pre_models = [None,None]

def print_summary_log(logger, trn_acc_glb, val_acc_glb, tst_acc_glb, trn_acc_obj, val_acc_obj, tst_acc_obj):
  logger.info('')

  logger.info('global-level model: ' + str(glb.model_name))
  logger.info('pretrained: ' + str(glb.use_pretrained_params))
  logger.info('fine tune all layers: ' + str(glb.fine_tune_all_layers))
  logger.info('epochs: ' + str(glb.num_epochs))
  logger.info('batch size: ' + str(train_batch_size))
  logger.info('learning rate: ' + str(glb.learning_rate))
  logger.info('prediction accuracy: %.4f%%, %.4f%%, %.4f%%' % (trn_acc_glb, val_acc_glb, tst_acc_glb))

  logger.info('')

  logger.info('object-level model: ' + str(obj.model_name))
  logger.info('pretrained: ' + str(obj.use_pretrained_params))
  logger.info('fine tune all layers: ' + str(obj.fine_tune_all_layers))
  logger.info('epochs: ' + str(obj.num_epochs))
  logger.info('batch size: ' + str(train_batch_size))
  logger.info('learning rate: ' + str(obj.learning_rate))
  logger.info('prediction accuracy: %.4f%%, %.4f%%, %.4f%%' % (trn_acc_obj, val_acc_obj, tst_acc_obj))


def evaluate(logger, models, train_loaders, validation_loaders, test_loaders):
  logger.info('')
  logger.info('evaluating model on multiple sets combining both global-level and object-level models\' predictions')
  logger.info('predict weights: ' + str(predict_weights[0]) + ', ' + str(predict_weights[1]))
  begin_time = time.time()

  helper.evaluate(
    logger=logger,
    models=models,
    data_loaders=train_loaders,
    set_name='train set',
    predict_weights=predict_weights
  )
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

  logger.info('evaluation has been done! total time: %.4fs' % (time.time() - begin_time))


def get_model_with_saved_parameters(model_path_glb, model_path_obj):
  model_glb = helper.get_model_by_name(glb.model_name, pretrained=False)
  helper.replace_model_fc(glb.model_name, model_glb)
  model_glb.load_state_dict(torch.load(model_path_glb))
  model_glb = model_glb.cuda()

  model_obj = helper.get_model_by_name(obj.model_name, pretrained=False)
  helper.replace_model_fc(obj.model_name, model_obj)
  model_obj.load_state_dict(torch.load(model_path_obj))
  model_obj = model_obj.cuda()

  return model_glb, model_obj


def run_on_single_gpu(logger, data_loaders_glb, data_loaders_obj,
                      train_loaders, valid_loaders, test_loaders, pre_models, fine_tune_all_layers, num_epochs):
  # if you want to change hyper-parameters like number of epochs or learning rate for each level's training,
  # please go to corresponding module file
  _, trn_acc_glb, val_acc_glb, tst_acc_glb, model_path_glb = glb.get_trained_model_global(
    logger=logger, data_loaders=data_loaders_glb, train_batch_size=train_batch_size,
    save_model=True, pre_model=pre_models[0], fine_tune_all_layers=fine_tune_all_layers, num_epochs=num_epochs)
  _, trn_acc_obj, val_acc_obj, tst_acc_obj, model_path_obj = obj.get_trained_model_object(
    logger=logger, data_loaders=data_loaders_obj, train_batch_size=train_batch_size,
    save_model=True, pre_model=pre_models[1], fine_tune_all_layers=fine_tune_all_layers, num_epochs=num_epochs)

  print_summary_log(logger, trn_acc_glb, val_acc_glb, tst_acc_glb, trn_acc_obj, val_acc_obj, tst_acc_obj)
  model_glb, model_obj = get_model_with_saved_parameters(model_path_glb, model_path_obj)
  evaluate(
    logger=logger,
    models=[model_glb, model_obj],
    train_loaders=train_loaders,
    validation_loaders=valid_loaders,
    test_loaders=test_loaders
  )
  return model_glb, model_obj

def target_model_global(q_glb, data_loaders_glb, pre_model, fine_tune_all_layers, num_epochs):
  logger_glb = glb.get_logger(train_batch_size, add_console_log_prefix=True)
  logger_glb.info('target model global starts')
  _, trn_acc_glb, val_acc_glb, tst_acc_glb, model_path_glb = glb.get_trained_model_global(
    logger=logger_glb, data_loaders=data_loaders_glb, train_batch_size=train_batch_size,
    cuda_device_idx=0, save_model=True, pre_model=pre_model, fine_tune_all_layers=fine_tune_all_layers,
    num_epochs=num_epochs )
  q_glb.put(trn_acc_glb)
  q_glb.put(val_acc_glb)
  q_glb.put(tst_acc_glb)
  q_glb.put(model_path_glb)
  logger_glb.info('target model global stops')

def target_model_object(q_obj, data_loaders_obj, pre_model, fine_tune_all_layers, num_epochs):
  logger_obj = obj.get_logger(train_batch_size, add_console_log_prefix=True)
  logger_obj.info('target model object starts')
  _, trn_acc_obj, val_acc_obj, tst_acc_obj, model_path_obj = obj.get_trained_model_object(
    logger=logger_obj, data_loaders=data_loaders_obj, train_batch_size=train_batch_size,
    cuda_device_idx=1, save_model=True, pre_model=pre_model,fine_tune_all_layers=fine_tune_all_layers,
    num_epochs=num_epochs )
  q_obj.put(trn_acc_obj)
  q_obj.put(val_acc_obj)
  q_obj.put(tst_acc_obj)
  q_obj.put(model_path_obj)
  logger_obj.info('target model object stops')

def run_on_multiple_gpus(logger, data_loaders_glb, data_loaders_obj,
                         train_loaders, valid_loaders, test_loaders, pre_models, fine_tune_all_layers, num_epochs):

  q_glb = mp.Queue()  # store models and accuracies
  q_obj = mp.Queue()
  process_glb = mp.Process(target=target_model_global,
                           args=(q_glb, data_loaders_glb, pre_models[0], fine_tune_all_layers, num_epochs,))
  process_obj = mp.Process(target=target_model_object,
                           args=(q_obj, data_loaders_obj, pre_models[1], fine_tune_all_layers, num_epochs,))

  process_glb.start()
  process_obj.start()

  process_glb.join()  # join current process(main process), then current process will stop until process_glb finishes
  process_obj.join()

  trn_acc_glb    = q_glb.get() # FIFO
  val_acc_glb    = q_glb.get()
  tst_acc_glb    = q_glb.get()
  model_path_glb = q_glb.get()

  trn_acc_obj    = q_obj.get()
  val_acc_obj    = q_obj.get()
  tst_acc_obj    = q_obj.get()
  model_path_obj = q_obj.get()

  print_summary_log(logger, trn_acc_glb, val_acc_glb, tst_acc_glb, trn_acc_obj, val_acc_obj, tst_acc_obj)
  model_glb, model_obj = get_model_with_saved_parameters(model_path_glb, model_path_obj)
  evaluate(
    logger=logger,
    models=[model_glb, model_obj],
    train_loaders=train_loaders,
    validation_loaders=valid_loaders,
    test_loaders=test_loaders
  )
  return model_glb, model_obj

if __name__ == "__main__":
  log_file_name_prefix = 'combined'
  logger = utils.get_logger(log_file_name_prefix)

  logger.info('start loading dataset')
  begin_time = time.time()
  train_loader_glb, valid_loader_glb = dataset.get_train_validation_data_loader(
    resize_size=224,  # apply random crop for train set
    batch_size=train_batch_size,
    random_seed=random_seed,
    augment=True,
    validation_size=validation_size,
    object_boxes_dict=None,
    show_sample=False
  )
  test_loader_glb = dataset.get_test_data_loader(
    resize_size=224,  # no any crop
    batch_size=test_batch_size,
    object_boxes_dict=None
  )

  bounding_boxes = utils.get_annotated_bounding_boxes()
  train_loader_obj, valid_loader_obj = dataset.get_train_validation_data_loader(
    resize_size=(224, 224),  # for object level model, we don't need cropping any more!
    batch_size=train_batch_size,
    random_seed=random_seed,
    augment=True,
    validation_size=validation_size,
    object_boxes_dict=bounding_boxes,
    show_sample=False
  )
  test_loader_obj = dataset.get_test_data_loader(
    resize_size=224,
    batch_size=test_batch_size,
    object_boxes_dict=bounding_boxes
  )
  logger.info('loading dataset costs %.4fs' % (time.time() - begin_time))

  data_loaders_glb = [train_loader_glb, valid_loader_glb, test_loader_glb]
  data_loaders_obj = [train_loader_obj, valid_loader_obj, test_loader_obj]

  train_loaders = [train_loader_glb, train_loader_obj]
  valid_loaders = [valid_loader_glb, valid_loader_obj]
  test_loaders = [test_loader_glb, test_loader_obj]
  pre_models = [None, None]

  # test: it seems ResNet is better for global model and DenseNet better for object-level model
  glb.model_name = 'resnet152'
  obj.model_name = 'densenet161'
  fine_tune_all_layers=False
  glb.use_multiple_gpu=False
  obj.use_multiple_gpu=False
  num_epochs = 160
  if not use_multiple_gpu:
    pre_models[0], pre_models[1] = run_on_single_gpu(logger, data_loaders_glb, data_loaders_obj,
                                                     train_loaders, valid_loaders, test_loaders, pre_models, fine_tune_all_layers, num_epochs)
  else:
    mp.set_start_method('spawn')  # CUDA requires this
    pre_models[0], pre_models[1] = run_on_multiple_gpus(logger, data_loaders_glb, data_loaders_obj,
                                                        train_loaders, valid_loaders, test_loaders, pre_models, fine_tune_all_layers, num_epochs)

  fine_tune_all_layers = True
  num_epochs = 120
  if not use_multiple_gpu:
    run_on_single_gpu(logger, data_loaders_glb, data_loaders_obj, train_loaders, valid_loaders, test_loaders, pre_models, fine_tune_all_layers, num_epochs)
  else:
    run_on_multiple_gpus(logger, data_loaders_glb, data_loaders_obj, train_loaders, valid_loaders, test_loaders, pre_models, fine_tune_all_layers, num_epochs)


