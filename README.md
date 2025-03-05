# nld-floorplan-segmentation

## **Project Overview:**

A deep learning-based tool for image segmentation of Naver Real Estate floor plan images. This repository implements semantic segmentation models to accurately identify and segment structural elements such as rooms, walls, doors, and windows from floor plan images. This can be used for automated floor plan analysis, 3D model generation, and real estate data processing.

## **Directory Structure:**

```
├── src/                          # Source code directory
│   ├── configs/                  # Configuration files
│   ├── datasets/                 # Dataset processing
│   ├── loss/                     # Loss functions
│   ├── models/                   # Model implementations
│   ├── utils/                    # Utility functions
├── logs/                         # Log directory
└── main.py                       # Main entry point
```

## **Key Features:**

- **Segmentation Models:** Utilizes the `segmentation_models_pytorch` library to provide various pre-built segmentation models.
- **Specialized Loss Functions:** Implements balanced entropy loss and class balanced loss for handling class imbalance in segmentation tasks.
- **Data Augmentation:** Various techniques to improve model robustness and generalization.
- **Training Utilities:** Learning rate adjustment, early stopping, logging, and visualization tools.
- **Performance Optimization:** CUDA acceleration and cuDNN benchmarking for improved training speed.

## **Running the Code:**

The `main.py` script serves as the primary entry point for training and evaluating segmentation models. It handles:

- **Setup:** Configuration loading, seed setting, and GPU initialization.
- **Model Initialization:** Constructs the segmentation model with MADGRAD optimizer and cosine annealing scheduler.
- **Data Handling:** Loads floor plan dataset with augmentation techniques.
- **Training Loop:** Iteratively trains the model with periodic validation.
- **Model Evaluation:** Tracks performance metrics (loss and mIoU) and saves improved models.
- **Visualization:** Generates GIFs to visualize training progress.

To run the code:

```bash
python main.py
```

## **Getting Started:**

### **Requirements**

- Python 3.7+
- PyTorch 1.7+
- CUDA-enabled GPU (recommended)
- `segmentation_models_pytorch`
- MADGRAD optimizer

### **Installation**

1. Clone the repository:

```bash
git clone https://github.com/Minhan-Bae/nld_floorplan_seg.git
cd nld_floorplan_seg
```

2. Install the necessary packages:

```bash
pip install -r requirements.txt
```

### **Data Preparation**

Prepare the Naver Real Estate floor plan images and corresponding segmentation masks, then specify their location in the configuration file.

### **Training**

1. Configure training parameters in `src/configs/default.py`
2. Run the main script:

```bash
python main.py
```

### **Evaluation**

```bash
python src/models/evaluate.py --model_path [model_path] --config_file src/configs/default.py
```

## **Training Process:**

The training pipeline includes:
- Model initialization with configurable architectures and backbones
- Data loading with augmentation
- Training with specialized loss functions
- Periodic validation and model checkpointing
- Early stopping to prevent overfitting
- Visualization of segmentation results

## **Contributing:**

Contributions are welcome! Please feel free to submit pull requests or open issues for bug fixes, feature improvements, or documentation.

## **License:**

[Project License Information]

## **Acknowledgments:**

- This project utilizes the `segmentation_models_pytorch` library
- Thanks to the developers of the MADGRAD optimizer
- Naver Real Estate for the floor plan dataset
