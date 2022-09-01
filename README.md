### Metric 
  - Verification score 평균

### SmoothSwap 

- Train
  - embedder 
      - `python train_id_emb.py --data_dir="{vggface Path}" --multi_gpu=True`
      - `python train_swapper.py --data_dir={DATA PATH} --checkpoint_dir="{WHERE TO SAVE CHECKPOINTS}"`


**FULL CONFIGS**
```yaml
train:
  isTrain: false
  end_epoch: 1000
  batch_size: 512
  optimizer: adam
  weight_decay: 0.0005
  momentum: 0.9
  lr: 0.001
  lr_step: 10
  log_dir: ./logs
  checkpoint_dir: ./checkpoints
  device: cpu
id_emb:
  train_embedder: false
  network: resnet50
  emb_size: 512
  checkpoint_path: ''
generator:
  image_size: 256
  num_feature_init: 64
discriminator:
  image_size: 256
authur:
  name: conor.k
  version: 0.0.1
```
