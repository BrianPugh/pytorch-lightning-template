
# Usage

### Install Requirements

```
pip install -r requirements.txt
```

### Download and Prepare Dataset
First, download the dataset via

```
python scripts/download_dataset.py
```

once the data is downloaded, run the preprocessing script:

```
python scripts/prepare_dataset.py
```

### Training
After the dataset has been downloaded and prepared, begin training:

```
python train.py
```

you can view tensorboard logs:

```
tensorboard --logdir=lightning_logs --port 6006 --host 0.0.0.0 --samples_per_plugin images=1000
```
