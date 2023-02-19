# SoftVC VITS Singing Voice Conversion
## English docs
[Check here](Eng_docs.md)

```
conda create -n neosovit python=3.10
conda activate neosovit
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -r requirements.txt
```

```
python .\inference_main.py --model "\models\kita\D_60000.pth" --config "\models\kita\config.json" --input "\raw\test2.wav" --hubert "\hubert\hubert-soft-0d54a1f4.pt"
```

```
python .\resample.py --in_dir "\dataset_raw" --out_dir2 "\dataset_raw\32"
```

```
python .\preprocess_flist_config.py --source_dir "\dataset_raw\32" --eval_interval 1000
```

```
python .\preprocess_hubert_f0.py --in_dir "\dataset_raw\32" 
```

Note: You may need to adjust the eval interval so that it can actually saves (Time under epoch time % eval_interval == 0 is the condition)
```
python .\train.py --model {model_name} --config "\logs\{model_name}\config.json" --batch 12 --workers 8
```