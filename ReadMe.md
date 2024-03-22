# Adaptive Outlier Optimization for Online Test-Time OOD Detection
This project is for the paper: AUTO: Adaptive Outlier Optimization for Online Test-Time OOD Detection.

## Required Packages

The following packages are required to be installed:

- [PyTorch](https://pytorch.org/)
- [Scipy](https://github.com/scipy/scipy)
- [Numpy](http://www.numpy.org/)
- [Sklearn](https://scikit-learn.org/stable/)

Our experiments are conducted on Ubuntu Linux 16.04 with Python 3.8.

## Datasets
### In-distribution and Auxiliary Outlier Datasets

- In-distribution training set:
  - [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html): included in PyTorch.
  - [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/index) Please download ImageNet-1k and place the training data and validation data in
`/data/imagenet/train` and  `/data/ood_data/ood_data_large_scale/Imagenet_val`, respectively.

* Auxiliary outlier training set:

  * [80 Million Tiny Images](https://groups.csail.mit.edu/vision/TinyImages/): to download **80 Million Tiny Images** dataset. After downloading it, place it in this directory: `/data/ood_data/ood_data_small_scale/80M_Tiny_Images`

### CIFAR OOD Datasets
We provide links and instructions to download each dataset in CIFAR benchmark:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `/data/ood_data/ood_data_small_scale/svhn`. Then run `python select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `/data/ood_data/ood_data_small_scale/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `/data/ood_data/ood_data_small_scale/places365/test_subset`. We provide the test list in `/CIFAR/data_pre/places365_test_list.txt`.
* [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `/data/ood_data/ood_data_small_scale/LSUN_C`.
* [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `/data/ood_data/ood_data_small_scale/LSUN_resize`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `/data/ood_data/ood_data_small_scale/iSUN`.

For example, run the following commands in the **root** directory to download **LSUN-C**:
```
cd /data/ood_data/ood_data_small_scale
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```

### ImageNet OOD Datasets
Following the common setting, we have curated 4 OOD datasets from 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf), 
and de-duplicated concepts overlapped with ImageNet-1k.

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset,
which can be download via the following links:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, we use the entire dataset, which can be downloaded from their
[original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Please put all downloaded OOD datasets into `/data/ood_data/ood_data_large_scale/`.

## Model and checkpoints
You can modify `/CIFAR/model_pre/model_loader.py` to load your model and use it with AUTO to enhance OOD detection performance.

For ImageNet models, our code will download pre-trained models automatically.

## Test-time OOD detection
Please modify `config.py` in `/CIFAR/` or `/ImageNet/` to set test scenarios

After setting `config.py`, run:

```
python memory_test.py
```




