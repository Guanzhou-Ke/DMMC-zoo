# DMSC
This is a non official implementation version of DMSC via `pytorch`.

This repository contains the implementation of the paper "Deep multimodal subspace clustering networks" by Mahdi Abavisani and Vishal M. Patel. The paper was posted on arXiv in May 2018.

"Deep multimodal subspace clustering networks" (DMSC)  investigated various fusion methods for the task of multimodal subspace clustering, and suggested a new fusion technique called "affinity fusion" as the idea of integrating complementary information from two modalities with respect to the similarities between datapoints across different modalities. 

**NOTE:** Since the `svds` spent a lot of memory, it is too slow during the training.

## Prerequest

* python 3.7+
* pytorch 1.7.0 (CUDA 10.1)



## Quick Start

You can also install the environment as the following:
```bash
conda create --name dmsc --file requirements.txt -c pytorch
conda activate dmsc
```

Then, use `unittest` to test this project, following:

```bash
cd tests
export PYTHONPATH="../src"
python -m unittest
```

Finally, you could run the `model.py` directly, and you will be see the result of the program.
```
cd ./src
python models.py
```

## Official Implementations of DMSC:

Tensorflow implementation of DMSC from @Mahdi's [github page](https://github.com/mahdiabavisani/Deep-multimodal-subspace-clustering-networks) 

## Citation

Please use the following to refer to this work in publications:

<pre><code>
@ARTICLE{8488484, 
author={M. {Abavisani} and V. M. {Patel}}, 
journal={IEEE Journal of Selected Topics in Signal Processing}, 
title={Deep Multimodal Subspace Clustering Networks}, 
year={2018}, 
volume={12}, 
number={6}, 
pages={1601-1614}, 
doi={10.1109/JSTSP.2018.2875385}, 
ISSN={1932-4553}, 
month={Dec},}
</code></pre>