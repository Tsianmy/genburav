# GenbuRav (Genbu Ravine)

Learning Pytorch.

## Installation

```
conda env create -f environment.yml
```

or

```
pixi install
```

## Quick Start

### Prepare CIFAR10

Download CIFAR10 and make the directory under `data/` as below.

```
data/cifar10/
`-- cifar-10-batches-py/
    |-- batches.meta
    |-- data_batch_1
    |-- data_batch_2
    |-- data_batch_3
    |-- data_batch_4
    |-- data_batch_5
    |-- readme.html
    `-- test_batch
```

### Train

```
python run.py train --devs 0 --cfg configs/simnet_cifar10.yaml
```

### Test

```
python run.py test --devs 0 --cfg outputs/simnet_cifar10/xxxxxx_xxxxxx/simnet_cifar10.yaml --checkpoint outputs/simnet_cifar10/xxxxxx_xxxxxx/latest.ckpt
```

## Usage

### Train

```
python run.py train --devices <DEVICES> --cfg <CONFIG PATH> [--output_dir <OUTPUT>] [--search_opts <OPTIONS> [<OPTIONS> ...]] [--resume <CHECKPOINT PATH>] [--log_freq <LOG FREQUENCY>]
```

or

```
pixi run train ...
```

#### Example

```
python run.py train --devices 0,1 --cfg configs/simnet_cifar10.yaml --log_freq 3
```

Search hyperparameters

```
python run.py train --devices 0,1 --cfg configs/simnet_cifar10.yaml --search_opts 'optimizer.lr=[0.05,0.1]' 'train_transforms[0].padding=2' 'data_preprocessor.mean=[[128,125,112],[129.304,124.070,112.434]]'
```

Resume from checkpoint

```
python run.py train --devices 0,1 --cfg outputs/x/simnet_cifar10.yaml --resume outputs/x/latest.ckpt
```

### Test

```
python run.py test --devices <DEVICES> --cfg <CONFIG PATH> --checkpoint <CHECKPOINT PATH> [--log_freq <LOG FREQUENCY>]
```

or

```
pixi run test ...
```

#### Example

```
python run.py test --devices 0,1 --cfg outputs/x/simnet_cifar10.yaml --checkpoint outputs/x/latest.ckpt --log_freq 3
```

## Acknowledgements

* [OpenMMLab projects](https://github.com/open-mmlab)
* [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
* [Neural Point-Based Graphics](https://github.com/alievk/npbg)