├── .git/
├── jupyter/
│   ├── experiment_note.ipynb
    └── playground.ipynb
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
│   ├── models/
│   │   ├── loss/
│   │   │   ├── __init__.py
│   │   │   ├── balanced_entropy.py
│   │       └── class_balanced_loss.py
│   │   ├── __init__.py
│   │   ├── dfp_net.py
│   │   ├── metric.py
│   │   ├── model_test.py
│   │   ├── optimizer_manager.py
│   │   ├── segmentation_model.py
│       └── sgd_nan.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── adjust_lr.py
│   │   ├── arg_parser.py
│   │   ├── average_meter.py
│   │   ├── early_stopping.py
│   │   ├── logger.py
│   │   ├── seed_utils.py
│   │   ├── sgd_nan_handler.py
│   │   ├── str_to_bool.py
│   │   ├── trainer.py
│       └── visualize_instance_segmentation.py
│   ├── __init__.py
    └── main.py
│   ├── README.md
│   ├── __init__.py
│   ├── directory_structure.md
    └── requirements.txt