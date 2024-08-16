cd DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev

python3 simsiam.py \
  --num_workers 20 \
  --dataset_name Model_II \
  --train_config CvT \
  --batch_size 64 \
  --epochs_pretrain 15 \
  --epochs_finetune 20 \
  --cuda 

python3 simsiam.py \
  --num_workers 20 \
  --dataset_name Model_II \
  --train_config CrossFormer \
  --batch_size 64 \
  --epochs_pretrain 15 \
  --epochs_finetune 20 \
  --cuda 

python3 simsiam.py \
  --num_workers 20 \
  --dataset_name Model_II \
  --train_config LeViT \
  --batch_size 64 \
  --epochs_pretrain 15 \
  --epochs_finetune 20 \
  --cuda 

python3 simsiam.py \
  --num_workers 20 \
  --dataset_name Model_II \
  --train_config TwinsSVT \
  --batch_size 64 \
  --epochs_pretrain 15 \
  --epochs_finetune 20 \
  --cuda 