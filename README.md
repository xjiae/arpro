## Create an environment and install requirements:

`pip install -r requirements.txt `

## Download dataset
SWaT: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
VisA: https://github.com/HuiZhang0812/DiffusionAD?tab=readme-ov-file

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
