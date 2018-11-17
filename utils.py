import sys, os, time
import logging
import requests
import tarfile
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def download_file(url, filename):
  print('Downloading ' + filename + ' from ' + url)
  with open(filename, 'wb') as file:
    resp = requests.get(url, stream=True)
    # file.write(resp.content)
    # reference: https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    total_length = resp.headers.get('content-length')
    if total_length is None:  # no content length header
      file.write(resp.content)
    else:
      dl = 0
      total_length = int(total_length)
      for data in resp.iter_content(chunk_size=4096):
        file.write(data)
        dl += len(data)
        done = int(50 * dl / total_length)
        sys.stdout.write("\r[%s%s] %d%%" % ('=' * done, ' ' * (50 - done), done * 2))
        sys.stdout.flush()
  print()
  print(filename + ' has been downloaded successfully!')


def extract_tgz(filename):
  print('Extracting ' + filename + ' ...')
  tar = tarfile.open(filename, 'r:gz')
  tar.extractall()
  tar.close()
  print(filename + ' has been extracted successfully!')


def plot_images(class_names, images, classes_true, classes_pred=None):
  """
  Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
  """
  fig, axes = plt.subplots(3, 3)
  for i, ax in enumerate(axes.flat):
    # plot img
    ax.imshow(images[i, :, :, :], interpolation='spline16')
    # show true & predicted classes
    cls_true_name = class_names[classes_true[i]]
    if classes_pred is None:
      xlabel = "{0} ({1})".format(cls_true_name, classes_true[i])
    else:
      cls_pred_name = class_names[classes_pred[i]]
      xlabel = "True: {0}\nPred: {1}".format(
        cls_true_name, cls_pred_name
      )
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_yticks([])
  plt.show()


def get_annotated_bounding_boxes():
  fp = 'CUB_200_2011/bounding_boxes.txt'
  boxes = {}
  with open(fp, 'r') as file:
    for line in file:
      arr = line.split(' ')
      boxes[int(arr[0])] = (float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4]))
  return boxes


# deprecated
def get_logging(log_file_name_prefix):
  if not os.path.exists('logs/'):
    os.makedirs('logs/')
  time_str = time.strftime("%m-%d-%H-%M", time.localtime())
  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(message)s',
                      datefmt='%m-%d %H:%M',
                      filename='logs/' + log_file_name_prefix + '_' + time_str + '.log')
  # define a Handler which writes INFO messages or higher to the sys.stderr
  # console = logging.StreamHandler()
  # console.setLevel(logging.DEBUG)
  # # set a format which is simpler for console use
  # formatter = logging.Formatter('%(message)s')
  # # tell the handler to use this format
  # console.setFormatter(formatter)
  # # add the handler to the root logger
  # logging.getLogger('').addHandler(console)
  return logging


class LoggerS:  # Logger S, Logger Plus, Logger X, Logger X Plus~

  def __init__(self, logging, console_msg_prefix=None):
    self.logging = logging
    self.console_msg_prefix = console_msg_prefix

  def info(self, msg):  # i stands for info
    self.logging.info(msg)
    if self.console_msg_prefix is None:
      print(msg)
    else:
      print(self.console_msg_prefix + ' -> ' + msg)


def get_logger(log_file_name_prefix, console_msg_prefix=None):
  return LoggerS(get_logging(log_file_name_prefix), console_msg_prefix)