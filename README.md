## Create an environment and install requirements:

`pip install -r requirements.txt `

## Download datasets
SWaT: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

VisA: https://github.com/HuiZhang0812/DiffusionAD?tab=readme-ov-file


## Train models
Create a `_dump/` folder to store trained models and results.

Train vision anomaly detector:
`python train_ad.py --model fastflow --lr 0.0001 --epochs 500 --batch_size 8 --dataset visa --image_size 512 --category $category` 

Train time-series anomaly detector:
`python train_ad.py --model gpt2 --lr 0.0001 --epochs 500 --dataset swat --batch_size 16` 

Train DDPM diffusion models:
`python train_fixer.py --model diffusion --dataset visa --category $category --image_size 512 --num_epochs 100 --batch_size 2`

Train Diffusion_TS models:
`python train_fixer.py --model ts_diffusion --dataset swat --batch_size 16 --num_epochs 100`


## Run experiments

### RQ1 command:
For image data:
`python run_eval.py --task exp1 --dataset visa --category $category$ --image_size 512 --batch_size 1 --step 1000`

For time-series data:
`python run_eval.py --task exp1 --dataset swat --batch_size 8 --noise 100 --steps 500`

### RQ2 command:
For image data:
`python run_eval.py --task exp2 --dataset visa --category $category$ --image_size 512 --batch_size 1`

For time-series data:
`python run_eval.py --task exp2 --dataset swat --batch_size 8`
