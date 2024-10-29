#!/bin/bash

# MVTEC_CATEGORIES=("carpet" "pill" "hazelnut" "grid" "leather" "metal_nut" "screw" "tile" "toothbrush" "wood" "zipper" "bottle" "transistor" "cable" "capsule" )

MVTEC_CATEGORIES=("grid" "screw" "zipper")
# Loop through each category and run the training command
for category in "${MVTEC_CATEGORIES[@]}"
do
  echo "Starting evaluating for category: $category"
  # CUDA_VISIBLE_DEVICES=2 python train_ad.py --model fastflow --lr 0.0001 --epochs 500 --batch_size 8 --dataset visa --image_size 512 --category $category
#   CUDA_VISIBLE_DEVICES=2 python train_fixer.py --model diffusion --dataset mvtec --category $category --image_size 256 --num_epochs 100 --batch_size 2
   CUDA_VISIBLE_DEVICES=2 python run_eval.py --task exp1 --ad fastflow --dataset mvtec --category $category --image_size 256 --batch_size 8 --end 0.01
  echo "Finished testing for category: $category"
done

echo "Evaluation complete for all categories."
