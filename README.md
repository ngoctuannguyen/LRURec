# Enhance Sequential Recommendation via Linear Recurrent Units

This repository is the PyTorch implementation for SoICT 2025 paper:

**Enhance Sequential Recommendation via Linear Recurrent Units [][[Code](https://github.com/ngoctuannguyen/PLRec)]** (BibTex citation at the bottom)


## Requirements

Numpy, pandas, pytorch etc. For our detailed running environment see requirements.txt


## How to run PLRec
The command below specifies the training of LRURec on MovieLens-1M.
```bash
python train.py --dataset_code=ml-1m --CP_loss_weight=0.6
```

Excecute the above command (with arguments) to train LRURec, select dataset_code from ml-1m, beauty, video, sports, steam and xlong. XLong must be downloaded separately and put under ./data/xlong for experiments. Once trainin is finished, evaluation is automatically performed with models and results saved in ./experiments.


## Citation
Please consider citing the following paper if you use our methods in your research:



