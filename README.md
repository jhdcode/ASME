# Adaptive Structural Modality Enhancement in Score Calculation for Knowledge Graph Completion

## Usage

```bash
python run_gumbel.py \
  --do_train \
  --do_valid \
  --do_test \
  --data_path=data/MMKB-DB15K \
  --model=PairRE \
  -n=64 \
  -d=200 \
  -g=6 \
  -a=0.5 \
  -r=0.0 \
  -lr=0.0001 \
  -sns_lr=0.0001 \
  --sample_method=gumbel \
  --pre_sample_num=1500 \
  --loss_rate=100 \
  --exploration_temp=10 \
  --gpu=0 \
  --max_steps=100000 \
  --valid_steps=10000 \
  -b=400 \
  -dr
```

## Acknowledgements

Thanks to [MMRNS](https://github.com/quqxui/MMRNS) for the codebase.
