Author: Hang'rui Bi (hangruibi@outlook.com)

Thie repository provides high-level neural network layers for constructing SOTA E3-equivariant neural networks based on [e3nn](https://e3nn.org/).
It also provides examples for several applications, such as potential energy surface modeling, dipole prediction, hamiltonian prediction, as well as [score-based](https://yang-song.github.io/blog/2021/score/) conformation generation.
Some codes are modified from [Nequip](https://github.com/mir-group/nequip).
The notebook for infer matrix transformation from data is modified from scripts written by Mario Geiger.
The script `sde.py` is modified from https://github.com/yang-song/score_sde_pytorch.

# Installation
Once the environment is prepared, you can run this repository. There is no need to install this repository itself.

## Using pip
You may prepare the environment by running `pip install -r requirements.txt`.
Or execute the following commands after installing python3.8.
```
pip install --upgrade pip wheel setuptools
pip install torch==1.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html

TORCH=1.11.0
CUDA=11.5
pip install --upgrade --force-reinstall https://data.pyg.org/whl/torch-1.11.0%2Bcu115/torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
pip install --upgrade --force-reinstall https://data.pyg.org/whl/torch-1.11.0%2Bcu115/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install --upgrade --force-reinstall https://data.pyg.org/whl/torch-1.11.0%2Bcu115/torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl
pip install --upgrade --force-reinstall https://data.pyg.org/whl/torch-1.11.0%2Bcu115/torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
pip install torch-geometric 

pip install e3nn wandb jupyterlab absl-py ml-collections ase h5py torch_runstats torch-ema
```
## Using Docker or Singularity
You may pull the image from docker hub `docker pull dahubao/geometric:e3nn_py38_cu115_torch11.0`, or build a docker image with the following command.

```
# python 3.8 cuda 11.5 pytorch 1.11.0

docker run -it --name geometric --gpus all --network host nvidia/cuda:11.5.0-cudnn8-devel-ubuntu20.04 bash

# in the container, run:
apt update
apt install --assume-yes python3 python3-pip tmux
# execute the commands in the section #using pip to install the python packages 

```

The image can also be used by [singularity](https://singularity-userdoc.readthedocs.io/en/latest/).

# Run
Notice that the command line args only contains those that do not affect the expected experiment result.
For example, the address to bind and the seed for the RNG.
The hyperparameters of the experiment, such as learning rate and model structure, should be managed by git, and are therefore provided in the config file instead.
Before running, prepare the datasets (refer to the [examples](#examples) section for examples) and put them in the path specified in the config file.
## Training
```
# if you are using a container
# docker run -it --network host -v ~:/root --name <image> --gpus all image  bash 
# or
# singularity shell --nv <image>

python3  train.py --config <name_of_the_config>

```
Notice that the argument `<name_of_the_config>` is *not* a filename, you should run `python3  train.py --config config_energy` instead of `python3  train.py --config e3_layers/configs/config_energy.py`.
The list of available configs are registered in `e3_layers/configs/__init__.py`.
You may run `python3 train.py --help` to inspect all command line args.
## Inference

```
python3 inference.py --config <name_of_the_config> --model_path <model_path> --output_keys <key1, key2,...> --output_path <output_path>
```

# <a name='examples'>Examples</a>
You can download the processed data from https://drive.google.com/drive/folders/1tXW7LabtOJapgs-AZFMhnI46do0-YS30?usp=sharing.
## Potential Energy Surface Modeling
Run `python3 train.py --config config_energy`. 
Jointly training on energy and force is also supported.
The accuracy is comparable to [Nequip](https://github.com/mir-group/nequip) on [QM9](https://data.dgl.ai/dataset/qm9_edge.npz) (total energy MAE 5.0 meV).
## Atomic Multiple Prediction
Run `python3 train.py --config config_dipole`. 
The accuracy is comparable to or better than [this paper](https://pubs.acs.org/doi/10.1021/acs.jctc.1c01021) on the dataset it used (dipole MAE 4.1e-4 ev\*A).
For predicting multipoles of higher degrees, you should decompose them into irreducible representations first.
## Hamiltonian Prediction
Run `python3 train.py --config config_hamiltonian`. 
Notice that this config only works for H2O computed in ORCA convention.
The accuracy is comprable (MAE 1.7e-5 Hatree) to [PhiSNet](https://proceedings.neurips.cc/paper/2021/hash/78f1893678afbeaa90b1fa01b9cfb860-Abstract.html) on the water dataset it used.
## Score Based Generative Model (continous Variance-Perserving-Stochastic-Differential-Equation)
First, clone https://github.com/yang-song/score_sde_pytorch and add it to your environment variable `PYTHONPATH`.
The prior for position is `randn(n_nodes, 3)`, so you need to set the argument `r_max` at least 7.
You may set up `data.std` in the config file to scale the input by `1/data.std`, such that the variance is similar before and after perturbation.
Run
```
python3 train_sde.py --sde_config ../score_sde_pytorch/configs/vp/cifar10_ncsnpp_continuous.py  --workdir results --e3_config config_diffusion  --wandb --wandb_project diffusion --config_spec embed_time_in_nodes --name test
```
It is normal that the loss does not decrease at all for the first few thousand steps.
The script assumes there is a tensor named `pos` and reserves the key `t` for time and `species` for molecule/atom/residual type.


# Model Interface and Data Format
The input and output of all modules in this repository are instances of `e3_layers.data.Data`.
A Data object is a dictionary of tensors that describes a graph or a point cloud.
For each tensor `data[key]`, there may be a correspoding `data.attrs[key] = (is_per, irreps)`.
The first item means whether the tensor is defined on graphs, edges, or nodes, while the second is the type of irreducible representation or the number of dimensions.
Such annotations help to infer the number of nodes and edges, perform equivariance tests, reshape the tensor, and select one item in a batch.
It is assumed that the shapes of tensors are `[cat_dim, channel_and_irreps]`.
The first dimension, cat_dim corresponds to the number of edges/nodes/graphs.
The second dimension equals to `irreps.dim`, e.g. a tensor of type `2x1e` is of dimension 6.

The class `e3_layers.data.Batch` inherits `e3_layers.data.Data`.
A batch object is multiple graphs considered as a single (disconnected) graph.
Samples are concatenated along their first dimension.
Special keys include `edge_index`, and `face`.
These indices are increased or decreased when collating into or selecting from a batch.
The advantage of doing so (instead of creating a batch dimension) is that no padding is ever needed.
It is both dictionary-like (indexed by keys) and list-like (indexed by (list of) integer indices).
The key `_n_graphs`, `_n_nodes`, `_node_segement`, `_n_edges`, and `_edge_segement` are reserved for separating the concatenated tensors since no padding is used.
The key `_rotation_matrix` is reserved for equivariance tests. Do not override them with keys in your dataset.

## Creating Datasets
The `e3_layers.data.CondensedDataset` inherits from `e3_layers.data.Batch`.
Refer to `data.ipynb` for the scripts to convert various data formats to the HDF5 format that this repository uses.

# Constructing Your Own Models
To add a new config, you can add a python file in the configs directory that defines `get_config()` and mention it in `configs/__init__.py`.
There are already some examples in the directory that you can refer to.
The `get_config` function should return an instance of `ml_collections.ConfigDict`.
You can fully customize the model architecture by specifying `config.model_config`.
The model config contains arguments for each layer in the model, and its structure determines the network module hierarchy.

# Modules Provided

## SequentialGraphNetwork
The wrapper class for composing layers together into neural networks.
Internally, it sequentially runs its layers and adds the output of each module to the dictionart of inputs. 
For most layers in this repository, the input keys and output keys can be customized during initialization, and the network topology is induced by matching keys.
In this way, this class can support a network topology of arbitrary DAG.

## Embedding

* OneHotEncoding
* RadialBasisEdgeEncoding
* SphericalEncoding. Can be used to generate edge attributes used in convolution, or to embed node orientations if they are not rotationally symmetric.

## Pointwise Operations

* PointwiseLinear: A wrapper of e3nn.o3.Linear, biases are only applied to scalars It mixes channels and preserves degree and parity.
* TensorProductExpansion, or [SelfMix](https://proceedings.neurips.cc/paper/2021/hash/78f1893678afbeaa90b1fa01b9cfb860-Abstract.html)   used to recombine (“mix”) the features different degrees and parities. This does not mix the channels and requires all irreps have the same number of channels.
* ResBlock

## Message Passing

### [FactorizedConvolution](https://github.com/mir-group/nequip)

A factorized convolution is usually much faster and parameter-economic than a fully connected one (e.g. these in TFN and SE3Transformer).
It internally calls tensor product with instruction `uvu` instead of `uvw`.
Linear layers are inserted before and after the tensor product to mix the channels.
By default, it uses a 'self-connection' layer, which typically contributes to most of its trainable parameters.
The self-connection layer performs a fully connected tensor product between the input features and the input embedding (e.g. one hot atom type encoding).

### MessagePassing

A message-passing layer consisting of a convolution, a non-linearity activation function, and optionally a residual connection.

## Output

* Pooling
* GradientOutput
* PerTypeScaleShift
* Pairwise: Constructs a pairwise representation (edge features) from pointwise representations (node features).
* TensorProductContraction: Constructs tensor product representations from irreps.


# Profiling
```
python3  train.py --config <name_of_the_config> --profiling
```
The results will be save as `<workdir>/<name>/profiling.json`. It is recommended to inspect the profile with chrome trace viewer. Saving the trace may cause a segfault, which is a known PyTorch [isssue](https://github.com/pytorch/pytorch/issues/69443).