# ETTh1
for seed in 1 2 3; do
  python train.py --dataset_path data/ETT/ETTh1.csv --features M  --seq_len 96 --pred_len 96 --e_layers 2 --d_model 256 --d_ff 256 --model itransformer --train_epoch 10 --seed $seed
done

for seed in 1 2 3; do
  python train.py --dataset_path data/ETT/ETTh1.csv --features M  --seq_len 96 --pred_len 192 --e_layers 2 --d_model 256 --d_ff 256 --model itransformer --train_epoch 10 --seed $seed
done

for seed in 1 2 3; do
  python train.py --dataset_path data/ETT/ETTh1.csv --features M  --seq_len 96 --pred_len 336 --e_layers 2 --d_model 512 --d_ff 512 --model itransformer --train_epoch 10 --seed $seed
done

# ETTh2
for seed in 1 2 3; do
  python train.py --dataset_path data/ETT/ETTh2.csv --features M  --seq_len 96 --pred_len 96 --e_layers 2 --d_model 256 --d_ff 256 --model itransformer --train_epoch 10 --seed $seed
done

for seed in 1 2 3; do
  python train.py --dataset_path data/ETT/ETTh2.csv --features M  --seq_len 96 --pred_len 192 --e_layers 2 --d_model 256 --d_ff 256 --model itransformer --train_epoch 10 --seed $seed
done

for seed in 1 2 3; do
  python train.py --dataset_path data/ETT/ETTh2.csv --features M  --seq_len 96 --pred_len 336 --e_layers 2 --d_model 512 --d_ff 512 --model itransformer --train_epoch 10 --seed $seed
done

# ETTm1
for seed in 1 2 3; do
  python train.py --dataset_path data/ETT/ETTm1.csv --features M  --seq_len 96 --pred_len 96 --e_layers 2 --d_model 128 --d_ff 128 --model itransformer --train_epoch 10 --seed $seed
done

for seed in 1 2 3; do
  python train.py --dataset_path data/ETT/ETTm1.csv --features M  --seq_len 96 --pred_len 192 --e_layers 2 --d_model 128 --d_ff 128 --model itransformer --train_epoch 10 --seed $seed
done

for seed in 1 2 3; do
  python train.py --dataset_path data/ETT/ETTm1.csv --features M  --seq_len 96 --pred_len 336 --e_layers 2 --d_model 128 --d_ff 128 --model itransformer --train_epoch 10 --seed $seed
done

# ETTm2
for seed in 1 2 3; do
  python train.py --dataset_path data/ETT/ETTm2.csv --features M  --seq_len 96 --pred_len 96 --e_layers 2 --d_model 128 --d_ff 128 --model itransformer --train_epoch 10 --seed $seed
done

for seed in 1 2 3; do
  python train.py --dataset_path data/ETT/ETTm2.csv --features M  --seq_len 96 --pred_len 192 --e_layers 2 --d_model 128 --d_ff 128 --model itransformer --train_epoch 10 --seed $seed
done

for seed in 1 2 3; do
  python train.py --dataset_path data/ETT/ETTm2.csv --features M  --seq_len 96 --pred_len 336 --e_layers 2 --d_model 128 --d_ff 128 --model itransformer --train_epoch 10 --seed $seed
done