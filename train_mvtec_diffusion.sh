#!/bin/bash

# Array of categories 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'fryum' 'macaroni1' 'macaroni2'  'capsules' 'chewinggum' 
# VISA_CATEGORIES=('candle' 'cashew' 'pipe_fryum' 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'fryum' 'macaroni1' 'macaroni2'  'capsules' 'chewinggum' )
# Array of MVTEC categories "bottle" "transistor" "cable" "capsule" "carpet"  "hazelnut" "leather" "metal_nut" "pill" "tile" "toothbrush" "wood" "grid" "screw" 
MVTEC_CATEGORIES=("zipper")

# Loop through each category and run the training command
for category in "${MVTEC_CATEGORIES[@]}"
do
  echo "Starting evaluating for category: $category"
  # CUDA_VISIBLE_DEVICES=2 python train_ad.py --model fastflow --lr 0.0001 --epochs 500 --batch_size 8 --dataset visa --image_size 512 --category $category
  CUDA_VISIBLE_DEVICES=2 python train_fixer.py --model diffusion --dataset mvtec --category $category --image_size 256 --num_epochs 100 --batch_size 2
  #  CUDA_VISIBLE_DEVICES=2 python run_eval.py --task exp2 --dataset visa --category $category --image_size 512 --batch_size 1 --noise 999 --end 10.0
  echo "Finished training for category: $category"
done

echo "Evaluation complete for all categories."
