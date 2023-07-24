An implementation of the Local Environment Pooling (LEPool) method [[`paper`](https://doi.org/10.1016/j.comptc.2023.114161)]. We only implemented LEPool for SchNet and GemNet-OC here, for demonstration purpose.


For SchNet, LEPool provides an improvement of ~0.04 eV in MAE value for the IS2RE task.
<div align="left">
    <img src="schnet.png" width=500px>
</div>

For GemNet-OC, LEPool provides an improvement of ~0.01 eV in MAE value for the IS2RE task.
<div align="left">
    <img src="gemnet_oc.png" width=500px>
</div>

# `ocp` by Open Catalyst Project

[![CircleCI](https://circleci.com/gh/Open-Catalyst-Project/ocp.svg?style=shield)](https://circleci.com/gh/Open-Catalyst-Project/ocp)
[![codecov](https://codecov.io/gh/Open-Catalyst-Project/ocp/branch/codecov/graph/badge.svg?token=M606LH5LK6)](https://codecov.io/gh/Open-Catalyst-Project/ocp)


`ocp` is the [Open Catalyst Project](https://opencatalystproject.org/)'s
library of state-of-the-art machine learning algorithms for catalysis.

<div align="left">
    <img src="https://user-images.githubusercontent.com/1156489/170388229-642c6619-dece-4c88-85ef-b46f4d5f1031.gif">
</div>

It provides training and evaluation code for tasks and models that take arbitrary
chemical structures as input to predict energies / forces / positions, and can
be used as a base scaffold for research projects. For an overview of tasks, data, and metrics, please read our papers:
 - [OC20](https://arxiv.org/abs/2010.09990)
 - [OC22](https://arxiv.org/abs/2206.08917)

Projects developed on `ocp`:

- CGCNN [[`arXiv`](https://arxiv.org/abs/1710.10324)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/cgcnn.py)]
- SchNet [[`arXiv`](https://arxiv.org/abs/1706.08566)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/schnet.py)]
- DimeNet [[`arXiv`](https://arxiv.org/abs/2003.03123)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/dimenet.py)]
- ForceNet [[`arXiv`](https://arxiv.org/abs/2103.01436)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/forcenet.py)]
- DimeNet++ [[`arXiv`](https://arxiv.org/abs/2011.14115)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/dimenet_plus_plus.py)]
- SpinConv [[`arXiv`](https://arxiv.org/abs/2106.09575)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/spinconv.py)]
- GemNet-dT [[`arXiv`](https://arxiv.org/abs/2106.08903)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/tree/main/ocpmodels/models/gemnet)]
- PaiNN [[`arXiv`](https://arxiv.org/abs/2102.03150)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/tree/main/ocpmodels/models/painn)]
- Graph Parallelism [[`arXiv`](https://arxiv.org/abs/2203.09697)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/tree/main/ocpmodels/models/gemnet_gp)]
- GemNet-OC [[`arXiv`](https://arxiv.org/abs/2204.02782)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/tree/main/ocpmodels/models/gemnet_oc)]
- SCN [[`arXiv`](https://arxiv.org/abs/2206.14331)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/tree/main/ocpmodels/models/scn)]
- eSCN [[`arXiv`](https://arxiv.org/abs/2302.03655)] [[`code`](https://github.com/Open-Catalyst-Project/ocp/tree/main/ocpmodels/models/escn)]

## Installation

See [installation instructions](https://github.com/Open-Catalyst-Project/ocp/blob/main/INSTALL.md).

## Download data

Dataset download links and instructions are in [DATASET.md](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md).

## Train and evaluate models

A detailed description of how to train and evaluate models, run ML-based
relaxations, and generate EvalAI submission files can be found in
[TRAIN.md](https://github.com/Open-Catalyst-Project/ocp/blob/main/TRAIN.md).

Our evaluation server is [hosted on EvalAI](https://eval.ai/web/challenges/challenge-page/712/overview).
Numbers (in papers, etc.) should be reported from the evaluation server.

Interactive tutorial notebooks can be found
[here](https://github.com/Open-Catalyst-Project/ocp/tree/main/tutorials) to
get familiar with various components of the codebase.

## Pretrained model weights

We provide several pretrained model weights for download
[here](https://github.com/Open-Catalyst-Project/ocp/blob/main/MODELS.md).

## Discussion

For all non-codebase related questions and to keep up-to-date with the latest OCP
announcements, please join the [discussion board](https://discuss.opencatalystproject.org/).

All code-related questions and issues should be posted directly on our
[issues page](https://github.com/Open-Catalyst-Project/ocp/issues).

## Acknowledgements

- This codebase was initially forked from [CGCNN](https://github.com/txie-93/cgcnn)
by [Tian Xie](http://txie.me), but has undergone significant changes since.
- A lot of engineering ideas have been borrowed from [github.com/facebookresearch/mmf](https://github.com/facebookresearch/mmf).
- The DimeNet++ implementation is based on the [author's Tensorflow implementation](https://github.com/klicperajo/dimenet) and the [DimeNet implementation in Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py).

## License

`ocp` is released under the [MIT](https://github.com/Open-Catalyst-Project/ocp/blob/main/LICENSE.md) license.

## Citing `ocp`

If you use this codebase in your work, please consider citing:

```bibtex
@article{ocp_dataset,
    author = {Chanussot*, Lowik and Das*, Abhishek and Goyal*, Siddharth and Lavril*, Thibaut and Shuaibi*, Muhammed and Riviere, Morgane and Tran, Kevin and Heras-Domingo, Javier and Ho, Caleb and Hu, Weihua and Palizhati, Aini and Sriram, Anuroop and Wood, Brandon and Yoon, Junwoong and Parikh, Devi and Zitnick, C. Lawrence and Ulissi, Zachary},
    title = {Open Catalyst 2020 (OC20) Dataset and Community Challenges},
    journal = {ACS Catalysis},
    year = {2021},
    doi = {10.1021/acscatal.0c04525},
}
```
