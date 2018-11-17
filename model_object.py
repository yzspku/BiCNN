import cub_200_2011 as dataset
import helper, utils
import time

model_name = 'densenet161'
use_pretrained_params = True
fine_tune_all_layers = False

num_epochs = 100
# use to generate same train/validation data splits
random_seed = 96
# we use a part of train set as validation set
validation_size = 0.15
learning_rate = 8e-4
weight_decay = 5e-4
eval_epoch_step = 4

use_gpu = True
cuda_device_idx=0
use_multiple_gpu = False


def get_logger(train_batch_size, add_console_log_prefix=False):
  log_file_name_prefix = 'obj_' + model_name
  if use_pretrained_params:
    log_file_name_prefix += '_prtrn'
    if fine_tune_all_layers:
      log_file_name_prefix += 'All'
  log_file_name_prefix += '_ep' + str(num_epochs) + '_bt' + str(train_batch_size) + '_' + str(learning_rate)
  if add_console_log_prefix:
    return utils.get_logger(log_file_name_prefix, 'obj_' + model_name)
  else:
    return utils.get_logger(log_file_name_prefix)


# in fact I don't like writing train_batch_size here...QAQ
def get_trained_model_object(logger, data_loaders, train_batch_size, cuda_device_idx=cuda_device_idx, save_model=True,
                             pre_model=None, fine_tune_all_layers=fine_tune_all_layers, num_epochs=num_epochs):
  # if pre_model is None:
  #   use_pretrained_params = True
  #   fine_tune_all_layers = False
  # else :
  #   use_pretrained_params = False
  #   fine_tune_all_layers = True

  return helper.train_and_evaluate(
    logger=logger,

    model_name=model_name,
    pre_model=pre_model,
    use_pretrained_params=use_pretrained_params,
    fine_tune_all_layers=fine_tune_all_layers,

    data_loaders=data_loaders,
    is_object_level=True,

    num_epochs=num_epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    train_batch_size=train_batch_size,
    eval_epoch_step=eval_epoch_step,

    use_gpu=use_gpu,
    cuda_device_idx=cuda_device_idx,
    use_multiple_gpu=use_multiple_gpu,

    save_model=save_model
  )


if __name__ == "__main__":
  train_batch_size = 32
  test_batch_size = 32

  logger = get_logger(train_batch_size)

  logger.info('start loading dataset')
  begin_time = time.time()
  bounding_boxes = utils.get_annotated_bounding_boxes()
  train_loader, valid_loader = dataset.get_train_validation_data_loader(
    resize_size=224,
    batch_size=train_batch_size,
    random_seed=random_seed,
    augment=True,
    validation_size=validation_size,
    object_boxes_dict=bounding_boxes,
    show_sample=False
  )
  test_loader = dataset.get_test_data_loader(
    resize_size=224,
    batch_size=test_batch_size,
    object_boxes_dict=bounding_boxes
  )
  logger.info('loading dataset costs %.4fs' % (time.time() - begin_time))

  get_trained_model_object(
    logger=logger,
    data_loaders=[train_loader, valid_loader, test_loader],
    train_batch_size=train_batch_size
  )

  logger.info('model: ' + model_name)
  logger.info('pretrained: ' + str(use_pretrained_params))
  logger.info('fine tune all layers: ' + str(fine_tune_all_layers))
  logger.info('epochs: ' + str(num_epochs))
  logger.info('batch size: ' + str(train_batch_size))
  logger.info('learning rate: ' + str(learning_rate))