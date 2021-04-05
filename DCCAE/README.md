# DCCAE-with-pytorch
This is a non official implementation version of DCCAE via `pytorch`.

**NOTE:** This project just implements DCCAE and only refers Deep CCA version by [Michaelvll](https://github.com/Michaelvll/DeepCCA) and [VahidooX](https://github.com/VahidooX/DeepCCA).

DCCAE is a non-linear version of CCA which uses Auto-encoder as the mapping functions instead of fully-connected network(used in DCCA). DCCAE is originally proposed in the following paper:

Weiran Wang, Raman Arora, Karen Livescu, and Jeff Bilmes. "[On Deep Multi-View Representation Learning.](http://proceedings.mlr.press/v37/wangb15.pdf)", ICML, 2015.


## Prerequest

* python 3.7+
* pytorch 1.7.0 (CUDA 10.1)
* pytorch-lightning 1.2.1


## Quick Start

You can also install the environment as the following:
```bash
conda create -n dccae
conda activate dccae
conda install --yes --file requirements.txt
```

Then, use `unittest` to test this project, following:

```bash
cd tests
export PYTHONPATH="../src"
python -m unittest
```

Finally, you could run the `run.py` directly, and you will be see the result of the program.
```
cd ./src
python run.py --data_dir <dataset> --batch_size 256 [--checkpoint_dir <cpdir>] [--early_stop] [--gpu]
```


## Dataset
The model is evaluated on a noisy version of MNIST dataset. I use the dataset built by @VahidooX which is exactly like the way it is introduced in the paper. The train/validation/test split is the original split of MNIST.

The dataset was large and could not get uploaded on GitHub. So it is uploaded on another server. You can download them from [noisymnist_view1.gz](https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz) and [noisymnist_view2.gz](https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz), or use the download_data.sh. (Thanks to @VahidooX)


## Other Implementations of DCCA/DCCAE:

Keras implementation of DCCA from @VahidooX's [github page](https://github.com/VahidooX) The following are the other implementations of DCCA in MATLAB and C++. These codes are written by the authors of the original paper:

Torch implementation of DCCA from [@MichaelVll & @Arminarj](https://github.com/Michaelvll/DeepCCA)

C++ implementation of DCCA from [Galen Andrew's website](https://homes.cs.washington.edu/~galen/)

MATLAB implementation of DCCA/DCCAE from [Weiran Wang's website](http://ttic.uchicago.edu/~wwang5/dccae.html)
