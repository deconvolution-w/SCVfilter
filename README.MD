# SCVfliter

## Installation

Download the source code from github.
```shell script
git clone 
cd SCVfliter
```

Create a python virtual environment and install the required packages. If your device is cuda available, you can choose to use torch with gpu.
```shell script
conda create -n SCVfliter python=3.7
conda activate SCVfliter
pip install -r requirements.txt
```

## Prepare data

### Download

* The dataset-II has been provided in this repository.
* Other data can load from  [GISAID Initiative (epicov.org)](https://platform.epicov.org/epi3/frontend) 

### Quick Start

```shell script
python api.py
```

## Parameters

|  parameter   | default | type  |
| :----------: | :-----: | :---: |
|    nclass    |    3    |  int  |
|  split_rate  |   0.7   | float |
|  batch_size  |    4    |  int  |
|      lr      |  5e-5   | float |
| weight_decay |  1e-6   | float |
|   epoches    |   15    |  int  |
|   dict_len   |    6    |  int  |
|  embed_size  |   20    |  int  |
| num_hiddens  |   100   |  int  |
|  num_layers  |    2    |  int  |
|    device    |  cuda   |  str  |



