# Cat vs. Non-Cat & Fine-Grained Image Classification Project

This project aims to build an image classification pipeline, covering everything from data preprocessing and augmentation, building a baseline Convolutional Neural Network (CNN), utilizing Transfer Learning to train an advanced model, to hyperparameter tuning, model evaluation, and metric analysis.

---

## 👥 Team Roles & Responsibilities

This project is collaboratively completed by **Person A (Data Engineer / Model Developer)** and **Person B (Augmentation Specialist / Hyperparameter Optimizer)**.

| Phase | Task Module | Person A (Data & Baseline Model) | Person B (Augmentation & Transfer Learning) |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Data Preprocessing & Augmentation** | Write scripts for data cleaning and splitting (Resize, Normalization, 70/15/15 split). | Introduce data augmentation techniques (Rotation, Flip, Brightness adjustment, etc.). |
| **Phase 2** | **Model Building** | Set up the development environment and build a lightweight baseline CNN model from scratch. | Load pre-trained models (MobileNetV2/ResNet50) for Transfer Learning. |
| **Phase 3** | **Training & Hyperparameter Tuning**| Run training loops, monitor Train/Val Loss to prevent overfitting. | Experiment with hyperparameter combinations (Learning Rate, Batch Size <= 32, Adam/SGD). |
| **Phase 4** | **Model Evaluation & Metrics** | Collaboratively test the final models and calculate Accuracy, Confusion Matrix, Precision, and Recall. | Collaboratively test the final models and calculate Accuracy, Confusion Matrix, Precision, and Recall. |

---

## 📂 Recommended Project Directory Structure

To better organize code and data, the following directory structure is adopted:

```text
bsaeline model/
├── data/                  # Raw and processed datasets
│   ├── raw/               # Original 4,000 images
│   ├── train/             # Training set (70%)
│   ├── val/               # Validation set (15%)
│   └── test/              # Testing set (15%)
├── src/                   # Source code directory
│   ├── preprocess.py      # Data preprocessing & augmentation script (Person A & B)
│   ├── dataset.py         # PyTorch custom dataset loader
│   ├── baseline_model.py  # Custom lightweight CNN architecture (Person A)
│   ├── transfer_model.py  # MobileNetV2 / ResNet50 Transfer Learning model (Person B)
│   ├── train.py           # Unified model training and validation script (Person A)
│   └── evaluate.py        # Model evaluation and Confusion Matrix generation (Person A & B)
├── notebooks/             # Jupyter notebooks for analysis and visualization
│   └── exploration.ipynb  # Data exploration and results analysis
├── requirements.txt       # List of environment dependencies
└── README.md              # Project documentation
```

---

## 🛠️ Detailed Phase Planning

### 🔄 Phase 1: Data Preprocessing & Augmentation
* **Goal**: Clean and prepare the 4,000 images, expanding the dataset via augmentation to prevent overfitting.
* **Tasks**:
  * Uniformly resize images to 224 x 224 pixels.
  * Normalize pixel values to the range [0, 1].
  * Randomly split the dataset into **70% Training, 15% Validation, and 15% Testing**.
  * Introduce data augmentation: Random Rotation, Horizontal/Vertical Flip, Brightness Adjustments, etc.

### 🏗️ Phase 2: Building the Baseline & Transfer Model
* **Goal**: Quickly establish a functional baseline model and introduce pre-trained models for comparison.
* **Tasks**:
  * Set up the deep learning environment (PyTorch recommended).
  * **Baseline Model (Person A)**: Construct a minimalist CNN (e.g., 3 Convolutional layers + MaxPooling + Fully Connected layer) as a baseline to ensure the entire pipeline runs smoothly without crashing.
  * **Transfer Learning Model (Person B)**: Load a pre-trained **MobileNetV2** or **ResNet50**. Freeze the initial feature extraction layers and only rebuild and train the final Classification Head.

### ⚡ Phase 3: Training & Hyperparameter Tuning
* **Goal**: Execute efficient training and optimize model architecture and parameters.
* **Tasks**:
  * Write complete training loops, including forward pass, loss calculation, backward pass, and optimizer updates.
  * Monitor and record Train and Validation Loss and Accuracy in real-time, plotting curves to diagnose Overfitting or Underfitting.
  * Conduct hyperparameter experiments:
    * **Learning Rate**: Try different initial learning rates or add a learning rate decay strategy.
    * **Batch Size**: Keep it at or below 32 to avoid Out-Of-Memory (OOM) errors on laptops.
    * **Optimizer**: Compare the convergence speed and final performance of Adam vs. SGD.

### 📊 Phase 4: Evaluation & Metrics
* **Goal**: Evaluate the final models on an independent test set and generate a detailed performance report.
* **Tasks**:
  * Test the trained Baseline and Transfer Learning models on the Testing Set.
  * Calculate evaluation metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**.
  * Plot a **Confusion Matrix** to visually analyze categories the model easily confuses (e.g., which non-cat images were misclassified as cats).

---

## 🏆 Final Project Results (Phase 1)

All tasks for Phase 1 of this project were successfully completed. The pipeline was end-to-end trained and evaluated using a 10% subset of the real cat and boat dataset (1500+ images), proving the immense advantage of transfer learning.

### Final Test Set Evaluation Comparison

| Model Type | Test Set Accuracy | Training Setup | Performance Details |
| :--- | :--- | :--- | :--- |
| **Baseline CNN (Person A)** | **97.37%** | 3 Epochs (CPU) | 6 Misclassifications (4 boats classified as cats, 2 cats as boats) |
| **Transfer Learning ResNet (Person B)** | **100.00% (Perfect)** | 3 Epochs (CPU) | **Perfect Classification!** 0 Misclassifications |

**Conclusion**: Even with a small amount of data, few training epochs, and limited computational resources (CPU only), **Transfer Learning** using a large pre-trained backbone to extract features still achieved a perfect score of 100%, far exceeding the simple convolutional network trained from scratch.

---

## 🐾 Current 5-Breed Classification Pipeline (Cloud GPU)

*This section outlines the latest training methodology for the 5-category cat breed classification task (Pallas, Ragdoll, Singapura, Persian, Sphynx).*

### 1. Training Methodology & Data Augmentation
To achieve high accuracy on fine-grained cat breed classification with a limited dataset, we employ **Transfer Learning** using a pretrained backbone (**ResNet50**) with its feature extraction layers fully unfrozen for **Fine-Tuning** (`freeze_backbone=False`). A custom 5-class `nn.Sequential` head with **Dropout(0.5)** is trained to classify the specific breeds while preventing overfitting.

The data pipeline incorporates the following augmentations:
- **Resize**: Uniformly scaled to `224x224`.
- **RandomRotation**: `degrees=15`.
- **RandomHorizontalFlip**.
- **ColorJitter**: `brightness=0.2`.
- **ImageNet Normalization**: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`. (Crucial for unlocking the pretrained backbone's full potential).

The model is trained using **CrossEntropyLoss** and the **Adam Optimizer** (lr=0.0001, weight_decay=1e-4) combined with a **ReduceLROnPlateau** scheduler for 15+ epochs.

### 2. Model Exporting & Deployment
**Where is the model saved?**
During training (`src/train.py`), the script automatically monitors the Validation Loss. Whenever the validation loss reaches a new minimum, the model's weights are immediately saved to your local working directory under **`models/best_transfer.pth`**. 

**How to export and use it?**
1. **Download**: If you trained on a cloud GPU (e.g., JupyterLab), simply right-click the `models/best_transfer.pth` file in the file browser and click **Download** to save it to your local machine.
2. **Inference**: Once downloaded, you can load this `.pth` weight file into the robotic vehicle's computer. The repository includes an `src/inference_video.py` script, which can load `best_transfer.pth` to perform real-time bounding-box inference and breed classification directly on a live camera feed.
