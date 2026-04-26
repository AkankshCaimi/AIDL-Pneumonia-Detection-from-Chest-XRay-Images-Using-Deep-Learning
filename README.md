# Pneumonia Detection from Chest X-Ray Images using Deep Learning

## Project Overview
Binary classification of chest X-ray images as **Normal** or **Pneumonia** using:
1. Custom CNN (4-block architecture, trained from scratch)
2. VGG16 Transfer Learning (2-phase: frozen base then fine-tuning)
3. Traditional ML baselines (Logistic Regression, SVM, Random Forest with hand-crafted features)

## Dataset
**Citation:** Kermany, D., Zhang, K. and Goldbaum, M. (2018) Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, V2. doi: 10.17632/rscbjbr9sj.2.

- The dataset (~1.1 GB) is **downloaded automatically** when the notebook is run.
- No manual download is required.

---

## Target Platform

| Spec | Value |
|------|-------|
| **Azure VM** | Standard_NC4as_T4_v3 |
| **GPU** | NVIDIA Tesla T4 (16 GB VRAM) |
| **Python** | 3.10+ |
| **Framework** | TensorFlow 2.x |

---

## How to Run

### Prerequisites
- The root project folder **must** be named **AIDL-Pneumonia-Detection-from-Chest-XRay-Images-Using-Deep-Learning**.
  This is the default name when cloning from the Git repository:

      git clone https://github.com/<username>/AIDL-Pneumonia-Detection-from-Chest-XRay-Images-Using-Deep-Learning.git
      cd AIDL-Pneumonia-Detection-from-Chest-XRay-Images-Using-Deep-Learning

### Step 1 --- Environment Setup
Ensure the following packages are available. The notebook will attempt to **auto-install** any missing packages, but you can also install them manually:

    pip install tensorflow numpy pandas matplotlib seaborn opencv-python-headless Pillow scikit-learn scikit-image mahotas tqdm shap

### Step 2 --- Run the Notebook
1. Open the notebook (.ipynb) in JupyterLab / Jupyter Notebook on the Azure VM.
2. Ensure you are inside the **AIDL-Pneumonia-Detection-from-Chest-XRay-Images-Using-Deep-Learning** directory.
3. Click **Run All** (or Kernel > Restart and Run All).
4. The notebook will automatically:
   - Verify and install dependencies (Section 0.1)
   - Download and extract the dataset programmatically (Section 0.5)
   - Preprocess data, train all models, evaluate, and generate all plots end-to-end.

### Step 3 --- View Outputs
- Trained models are saved to ./models/
- All plots, metrics CSVs, and figures are saved to ./results/

---

## Estimated Runtime

| Mode | EPOCHS_CUSTOM | EPOCHS_TRANSFER_P1 | EPOCHS_TRANSFER_P2 | Approx. Time (T4 GPU) |
|------|:---:|:---:|:---:|:---:|
| **Reproducibility (default)** | 2 | 2 | 2 | ~15 min |
| **Full training (original)** | 50 | 30 | 20 | ~1.5 hours |

---

## Switching Between Reproducibility and Full Training

The notebook ships with **reduced hyperparameters** for fast, reproducible runs.
All original values are preserved as comments in **Section 0.4 (Global Constants)**.

To run with **original (full) hyperparameters**, open Section 0.4 and swap the comments:

    # Current (Reproducibility mode):
    EPOCHS_CUSTOM = 2              # Reduced for reproducibility on Azure T4
    EPOCHS_TRANSFER_P1 = 2         # Reduced for reproducibility on Azure T4
    EPOCHS_TRANSFER_P2 = 2         # Reduced for reproducibility on Azure T4

    # To restore original values, replace the above with:
    EPOCHS_CUSTOM = 50             # Original: full training
    EPOCHS_TRANSFER_P1 = 30        # Original: full training
    EPOCHS_TRANSFER_P2 = 20        # Original: full training

Similarly, EarlyStopping and ReduceLROnPlateau patience values are reduced
throughout the notebook. Each instance has a comment indicating the original value, e.g.:

    patience=5,    # Reduced for reproducibility (original: 10)

---

## Notebook Structure

| Section | Description |
|---------|-------------|
| **0** | Environment setup, GPU verification, global constants, dataset download |
| **1** | Dataset loading, organization, and 85/15 train/val split |
| **2** | Exploratory Data Analysis (class distribution, image sizes, pixel intensities, average images) |
| **3** | Preprocessing and data pipeline (augmentation, generators, class weights) |
| **4** | Custom CNN --- build, train, evaluate |
| **5** | VGG16 Transfer Learning --- 2-phase training, evaluate |
| **6** | Traditional ML baselines (HOG/Haralick/LBP features + LR, SVM, RF) with SHAP explainability |
| **7** | Comprehensive comparison of all models (metrics, ROC curves, confusion matrices, compute cost) |
| **8** | Grad-CAM interpretability for both CNN models |
| **9** | Conclusions and summary |

---

## Output Files

### Models (./models/)

| File | Description |
|------|-------------|
| custom_cnn_best.keras | Best Custom CNN checkpoint (by val_auc) |
| custom_cnn_final.keras | Final Custom CNN after training |
| transfer_p1_best.keras | Best VGG16 checkpoint after Phase 1 |
| transfer_finetuned_best.keras | Best VGG16 checkpoint after Phase 2 |
| transfer_final.keras | Final VGG16 model after both phases |

### Results (./results/)

| File | Description |
|------|-------------|
| full_comparison.csv | All models metrics in one table |
| custom_cnn_metrics.csv | Custom CNN test metrics |
| transfer_metrics.csv | VGG16 test metrics |
| class_distribution.png | Class balance across splits |
| sample_images.png | Example X-rays from each class |
| size_distribution.png | Image dimension distributions |
| intensity_distribution.png | Pixel intensity analysis |
| average_images.png | Mean image per class plus difference |
| augmented_samples.png | Augmented training examples |
| custom_cnn_training.png | Custom CNN training curves |
| transfer_training.png | VGG16 training curves (Phase 1 + 2) |
| cm_custom_cnn.png | Custom CNN confusion matrix |
| cm_transfer.png | VGG16 confusion matrix |
| cm_traditional_ml.png | Traditional ML confusion matrices |
| feature_importance.png | Random Forest feature group importance |
| shap_summary.png | SHAP global feature importance |
| shap_individual.png | SHAP per-prediction explanations |
| comparison_bars.png | All models metrics bar chart |
| roc_comparison.png | ROC curves for all models |
| training_time.png | Computational cost comparison |
| all_confusion_matrices.png | Side-by-side confusion matrices |
| gradcam_analysis.png | Grad-CAM visualizations (VGG16) |
| gradcam_comparison.png | Grad-CAM Custom CNN vs VGG16 |

---

