# MTS_CAD
The implementation of CAD which is used in anomaly detection.

## Getting started
clone the repo
```bash
git clone https://github.com/dawnvince/MTS_CAD.git && cd MTS_CAD
```
## Environment
Our implementation uses pytorch-lightning framework with Pytorch version 1.12.0 and **Python 3.8+**. Conda is recommended to set environment.

Install by conda
```bash
conda create
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

Other dependency please refer to `requirement.txt`

## RUN 
First, please put datasets(SMD, SWaT, WADI) under dataset folder. (Or modify path in `gen_data.py`). SMD is available in https://github.com/NetManAIOps/OmniAnomaly; SWaT and WADI are available in https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/. SWaT and WADI needs to be transformed to csv file manually. 
**Attention**: We choose latest SWaT.A2_Dec 2015(Version 1) and WADI.A2.

### preprocess data
```
python gen_data.py
```

### SMD
Train and Test
```bash
./run_SMD.sh
```
Get F1 score
```bash
# get best F1 under point-adjustment
python get_score.py --dataset SMD
# get best F1 under k-th point-adjustment 
python get_score.py --dataset SMD --kth
```

### SWaT
Train and Test
```bash
./run_SWaT.sh
```
Get F1 score
```bash
python get_score.py --dataset SWaT
```

### WADI
Train and Test
```bash
./run_WADI.sh
```
Get F1 score
```bash
python get_score.py --dataset WADI
```

## Score and Label file
The original result files are saved in folder `%dataset/%data_name/`, including `y.npy`(orignal values in testset), `y_hat.npy`(predicted values in testset), `y_label.txt`(anomaly score of testsets)

## Run Your Own Data
### If the dataset contains multiple entities like SMD:
1. Add data preprocess function like gen_SMD in `gen_data.py`. **MaxMinScalar and clip is needed.**
2. Add your own dataset in the `main` function in `get_score.py` like SMD.
3. Write your own script like `run_SMD.sh`.

### If the dataset contains only one entities like SWaT:
1. Add data preprocess function like gen_SWaT in `gen_data.py`. **MaxMinScalar and clip is needed.**
2. Add your own dataset in the `main` function in `get_score.py` like SWaT.
3. Write your own script like `run_SWaT.sh`.

The other processes are just like above.
