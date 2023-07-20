# nld-floorplan-segmentation

```python
├── configs/
│   ├── __init__.py
└── default.py
├── datasets/
│   ├── __init__.py
│   ├── augmentation.py
│   ├── dataloader.py
│   ├── dataset.csv
    └── dataset.py
├── loss/
│   ├── __init__.py
│   ├── balanced_entropy.py
│   ├── class_balanced_loss.py
│   ├── losses.py
│   ├── optimizer_manager.py
│   ├── sgd_nan.py
    └── sgd_nan_handler.py
├── models/
│   ├── __init__.py
│   ├── evaluate.py
    └── smp_models.py
├── utils/
│   ├── __init__.py
│   ├── adjust_lr.py
│   ├── arg_parser.py
│   ├── average_meter.py
│   ├── early_stopping.py
│   ├── logger.py
│   ├── seed_utils.py
│   ├── str_to_bool.py
│   ├── trasform_gif.py
    └── visualize_instance_segmentation.py
│   ├── __init__.py
│   ├── importmod.py
    └── main.py
```
