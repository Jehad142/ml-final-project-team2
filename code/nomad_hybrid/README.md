# Usage:

Setup
```
cd ~/src/ml-final-project-team2/code
...or...
export PYTHONPATH=~/src/ml-final-project-team2/code
```

Train
```
python -m nomad_hybrid.cli \
  --csv /shared/data/nomad2018/train.csv \
  --xyz_dir /shared/data/nomad2018/train \
  --model hybrid \
  --epochs 30 \
  --save_path /shared/data/checkpoints/hybrid.pt
```

Prediction
```
python -m nomad_hybrid.cli \
  --predict \
  --test_csv /shared/data/nomad2018/test.csv \
  --xyz_dir /shared/data/nomad2018/test \
  --model hybrid \
  --load_path /shared/data/checkpoints/hybrid.pt \
  --output predictions.csv
```
