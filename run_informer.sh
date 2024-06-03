# ETTh1
for seed in 1 2 3; do
  python train.py --model informer --features M --seq_len 96 --label_len 96 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --dataset_path data/ETT/ETTh1.csv --seed $seed 
done

for seed in 1 2 3; do
  python train.py --model informer --features M --seq_len 96 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --attn prob --dataset_path data/ETT/ETTh1.csv --seed $seed 
done

for seed in 1 2 3; do
  python train.py --model informer --features M --seq_len 96 --label_len 96 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --dataset_path data/ETT/ETTh1.csv --seed $seed 
done

# ETTh2
for seed in 1 2 3; do
  python train.py --model informer --features M --seq_len 96 --label_len 96 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --dataset_path data/ETT/ETTh2.csv --seed $seed 
done

for seed in 1 2 3; do
  python train.py --model informer --features M --seq_len 96 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --attn prob --dataset_path data/ETT/ETTh2.csv --seed $seed 
done

for seed in 1 2 3; do
  python train.py --model informer --features M --seq_len 96 --label_len 96 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --dataset_path data/ETT/ETTh2.csv --seed $seed 
done

# ETTm1
for seed in 1 2 3; do
  python train.py --model informer --features M --seq_len 96 --label_len 96 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --dataset_path data/ETT/ETTm1.csv --seed $seed 
done

for seed in 1 2 3; do
  python train.py --model informer --features M --seq_len 96 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --attn prob --dataset_path data/ETT/ETTm1.csv --seed $seed 
done

for seed in 1 2 3; do
  python train.py --model informer --features M --seq_len 96 --label_len 96 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --dataset_path data/ETT/ETTm1.csv --seed $seed 
done

# ETTm2
for seed in 1 2 3; do
  python train.py --model informer --features M --seq_len 96 --label_len 96 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --dataset_path data/ETT/ETTm2.csv --seed $seed 
done

for seed in 1 2 3; do
  python train.py --model informer --features M --seq_len 96 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --attn prob --dataset_path data/ETT/ETTm2.csv --seed $seed 
done

for seed in 1 2 3; do
  python train.py --model informer --features M --seq_len 96 --label_len 96 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --dataset_path data/ETT/ETTm2.csv --seed $seed 
done