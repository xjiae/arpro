#!/bin/bash

# VISA_CATEGORIES=('pcb2' 'pcb3' 'pcb4' 'fryum' 'macaroni1' 'macaroni2'  'capsules' 'chewinggum' 'candle' 'cashew' 'pipe_fryum' 'pcb1' )
VISA_CATEGORIES=('capsules' 'macaroni2')
# Loop through each category and run the training command
for category in "${VISA_CATEGORIES[@]}"
do
  echo "Starting evaluating for category: $category"
  # CUDA_VISIBLE_DEVICES=2 python train_ad.py --model fastflow --lr 0.0001 --epochs 500 --batch_size 8 --dataset visa --image_size 512 --category $category
#   CUDA_VISIBLE_DEVICES=2 python train_fixer.py --model diffusion --dataset mvtec --category $category --image_size 256 --num_epochs 100 --batch_size 2
   CUDA_VISIBLE_DEVICES=0 python run_eval.py --task exp1 --ad efficientad --dataset visa --category $category --image_size 512 --batch_size 1 --end 0.01
  echo "Finished testing for category: $category"
done

echo "Evaluation complete for all categories."
