├── logs/
├── src/
│   ├── configs/
│   │   ├── __init__.py
│       └── default.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── augmentations.py
│   │   ├── collate_fn.py
│   │   ├── dataloader.py
│       └── datasets.py
│   ├── loss/
│   │   ├── __init__.py
│   │   ├── balanced_entropy.py
│   │   ├── class_balanced_loss.py
│   │   ├── losses.py
│   │   ├── optimizer_manager.py
│   │   ├── sgd_nan.py
│       └── sgd_nan_handler.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dfp_net.py
│   │   ├── metric.py
│   │   ├── model_test.py
│       └── segmentation_model.py
│   ├── runner/
│   │   ├── __init__.py
│       └── trainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── adjust_lr.py
│   │   ├── arg_parser.py
│   │   ├── average_meter.py
│   │   ├── early_stopping.py
│   │   ├── importmod.py
│   │   ├── logger.py
│   │   ├── seed_utils.py
│   │   ├── str_to_bool.py
│       └── visualize_instance_segmentation.py
│   ├── __init__.py
│   ├── main.py
    └── main_v2.py
│   ├── README.md
│   ├── __init__.py
│   ├── directory_structure.md
    └── requirements.txt