# GenbuRav (Genbu Ravine)

Pytorch learning.

## Installation

To be completed.

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
cd genburav
python run.py train --devs 0 --cfg configs/simnet_cifar10.yaml
```

### Test

```
cd genburav
python run.py test --devs 0 --cfg outputs/simnet_cifar10/xxxxxx_xxxxxx/simnet_cifar10.yaml --checkpoint outputs/simnet_cifar10/xxxxxx_xxxxxx/latest.ckpt
```

## Usage

### Train

`python run.py train --devices <DEVICES> --cfg <CONFIG PATH> [--output_dir <OUTPUT>] [--search_opts OPTIONS [OPTIONS ...]]`

#### Example

`python run.py train --devices 0,1 --cfg configs/simnet_cifar10.yaml`

Search hyperparameters

`python run.py train --devices 0,1 --cfg configs/simnet_cifar10.yaml --search_opts 'optimizer.lr=[0.02,0.05,0.1]' 'train_transforms[0].padding=2'`

### Test

`python run.py test --devices <DEVICES> --cfg <CONFIG PATH> --checkpoint <CHECKPOINT PATH>`

#### Example

`python run.py test --devices 0,1 --cfg outputs/simnet_cifar10/xxxxxx_xxxxxx/simnet_cifar10.yaml --checkpoint outputs/simnet_cifar10/xxxxxx_xxxxxx/latest.ckpt`