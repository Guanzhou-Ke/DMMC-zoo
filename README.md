# DMMC-zoo
This repos reproduced major variants of the Deep Multi-Modal Clustering algorithm via `pytorch`.


## Note

This is my first try to combine and reproduce the major variants of Deep Multi-Modal Clustering(DMMC) algorithms by `pytorch`. Because all of the DMMC algorithms were implemented on a different platform, such as `Tensorflow` or `pytorch`, it is challenging to do some experiments for the new one in the same environment. This repos will bring convenience to all of the researchers in DMMC.

## Environment

For the detail, sees `requirements.txt` from each subfolder.

- python 3.8
- pytorch 1.7.1 (cuda 10.1)
- pytorch-lightning 1.2.1

## Algorithms

- [x] [DCCA an DCCAE](./DCCAE)
- [x] [Deep Multimodal Subspace Clustering Networks(DMSC)](./DMSC)
- [ ] [Deep adversarial multi-view clustering network(DAMC)](./DAMC)
- [ ] [End-to-end adversarial-attention network for multi-modal clustering(EAMC)](./EAMC)