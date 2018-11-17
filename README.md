# FGC_CUB-200-2011

Fine Grained Image Classification on CUB-200-2011

## Environment

We suggest using [Anaconda](https://anaconda.org/) to create a virtual environment for this program. Visit official website or [here](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/) to download the installer(Hope there is a GUI and a browser on your deep learning server machine).

Create a new virtual environment:
`conda create -n pytorch python=3.6`

Activate the environment on MacOS/Linux:
`source activate pytorch`

On Windows:
`activate pytorch`

### Requirements

**Note:** We suggest using `pip` instead of `conda` to install following requirements **on Windows**. The reason is that if you choose to use conda to install something like PyTorch or numpy, in order to speed up computation, another 3 packages start with `mkl` will also be downloaded. However, these `mkl` packages have conflicts with `conda` on Windows and you just cannot run the program.

If you're using MacOS or Linux, just ignore the note and enjoy `conda`~

If you want to speed up package download, you can add Tsinghua's package repository for `conda`:
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```
Visit [清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) for more information.

#### PyTorch
Visit [Official Website](https://pytorch.org/), choose correct OS/PM/Python-version/CUDA-version to get install command. Please install both `pytorch` and `torchvision`.

If your download speed is too slow, you can also add Tsinghua's repository specially for installing pytorch:

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda install pytorch torchvision
```

#### requests
`pip install requests`, note that `conda` doesn't contain this package.

#### Other requirements
`conda install matplotlib pillow`

## Train and evaluate globally

```
cd FGC_CUB-200-2011
source activate pytorch
python global.py
```
