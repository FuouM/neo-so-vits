# SoftVC VITS Singing Voice Conversion
## English docs
[Check here](Eng_docs.md)

```
python .\inference_main.py --model "\models\kita\D_60000.pth" --config "\models\kita\config.json" --input "\raw\test2.wav" --hubert "\hubert\hubert-soft-0d54a1f4.pt"
```

```
python .\resample.py --in_dir "\dataset_raw" --out_dir2 "\dataset_raw\32"
```

```
python .\preprocess_flist_config.py --source_dir "\dataset_raw\32" 
```

```
python .\preprocess_hubert_f0.py --in_dir "\dataset_raw\32" 
```

```
python .\train.py --model {model_name} --config "\logs\{model_name}\config.json" --batch 12 --workers 8
```