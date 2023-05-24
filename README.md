# nld_floorplan_seg

## Data structure
```python
├── .git/
├── data/
├── src/
│   ├── configs/
│   │   ├── __init__.py
│       └── default.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── augmentations.py
│   │   ├── dataloader.py
│       └── datasets.py
│   ├── models/
│   │   ├── loss/
│   │   ├── __init__.py
│   │   ├── metric.py
│   │   ├── model_test.py
│   │   ├── optimizer_manager.py
│       └── segmentation_model.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── arg_parser.py
│   │   ├── average_meter.py
│   │   ├── generate_init_files.py
│   │   ├── generate_requirements_txt.py
│   │   ├── logger.py
│   │   ├── save_ckpt.py
│   │   ├── seed_utils.py
│   │   ├── sgd_nan_handler.py
│   │   ├── str_to_bool.py
│       └── visualize_instance_segmentation.py
    └── __init__.py
│   ├── README.md
│   ├── __init__.py
│   ├── directory_structure.md
│   ├── experiment_note.ipynb
│   ├── main.py
    └── requirements.txt
```