# Brain Tumor Classification Using CNN and Transfer Learning  

## Project Overview  
This project focuses on brain tumor classification using convolutional neural networks (CNNs) and transfer learning techniques. It involves developing custom CNN architectures and leveraging state-of-the-art pre-trained models such as Xception, DenseNet-169, and ResNet-50 to achieve high classification accuracy.  

The project includes thorough data preprocessing, experimentation with hyperparameters, and analysis of model performance through visualization and evaluation metrics.  

---

## Table of Contents  
1. [Dataset](#dataset)  
2. [Methodology](#methodology)  
   - [Data Preprocessing](#data-preprocessing)  
   - [Custom CNN Architectures](#custom-cnn-architectures)  
   - [Transfer Learning Models](#transfer-learning-models)  
   - [Training and Evaluation](#training-and-evaluation)  
3. [Results](#results)  
4. [Model Architectures](#model-architectures)  
   - [Custom Model 1](#custom-model-1)  
   - [Custom Model 2](#custom-model-2)  
5. [Visualization](#visualization)  
6. [Future Scope](#future-scope)  
7. [How to Use](#how-to-use)  

---

## Dataset  
The MRI brain tumor dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c/data). The dataset consists of labeled MRI scans categorized as tumor or non-tumor.  

- **Dataset Details**:  
A private collection of T1, contrast-enhanced T1, and T2 magnetic resonance images separated by brain tumor type.
Images without any type of marking or patient identification, interpreted by radiologists and provided for study purposes.
The images are separated by astrocytoma, carcinoma, ependymoma, ganglioglioma, germinoma, glioblastoma, granuloma, medulloblastoma, meningioma, neurocytoma, oligodendroglioma, papilloma, schwannoma and tuberculoma.

---

## Methodology  

### Data Preprocessing  
1. **Loading the Dataset**: The MRI dataset was loaded and analyzed to identify class distribution and image quality.  
2. **Data Augmentation**: Applied transformations such as rotation, flipping, and zooming to enhance dataset diversity and prevent overfitting.  
3. **Data Splitting**: Divided the dataset into training, validation, and testing sets (e.g., 70%-20%-10%).  

### Custom CNN Architectures  
1. Designed two custom CNN architectures to serve as baseline models.  
2. Each architecture was designed to progressively reduce spatial dimensions while extracting hierarchical features through convolutional and pooling layers.  
3. Regularization techniques like dropout and batch normalization were applied for better generalization.  

### Transfer Learning Models  
1. **Xception**: Leveraged its depthwise separable convolutions for efficient feature extraction.  
2. **DenseNet-169**: Utilized dense connectivity to promote feature reuse and alleviate gradient-related issues.  
3. **ResNet-50**: Incorporated residual connections to improve training efficiency for deeper layers.  
4. Pre-trained on ImageNet and fine-tuned using the MRI dataset.  

### Training and Evaluation  
1. Experimented with optimizers like Adam and SGD, varied learning rates, and batch sizes for optimal training.  
2. Used regularization techniques (dropout: 0.45-0.5, L2 regularization) to mitigate overfitting.  
3. Evaluated models on test data using metrics such as accuracy, precision, recall, F1-score, and confusion matrices.  

---

## Model Architectures  

### Custom Model 1  
A CNN designed to extract features efficiently through four convolutional-pooling blocks, followed by dense layers for classification.  
- **Key Features**:  
  - 4 Conv2D + MaxPooling2D layers  
  - Batch normalization and dropout for regularization  
  - Final Dense layer with 44 output nodes (for multi-class classification)  

### Custom Model 2  
A lightweight CNN architecture optimized for faster training with three convolutional-pooling blocks.  
- **Key Features**:  
  - 3 Conv2D + MaxPooling2D layers  
  - Flattening and a single Dense layer for classification  
  - Regularization techniques including batch normalization and dropout  



---

## Visualization  
- **Training vs. Validation Loss/Accuracy**: Graphs illustrate model performance across epochs.  
- **Confusion Matrix**: Displays model predictions compared to actual classes, highlighting classification accuracy for each category.  

---

## Future Scope  
1. Expand dataset size for better generalization.  
2. Experiment with ensemble methods combining predictions from multiple models.  
3. Implement advanced augmentation techniques like GAN-generated synthetic data.  
4. Extend the approach to multi-class tumor classification.  

---

## How to Use  
1. **Clone the Repository**:  
   ```bash  
   git clone https://github.com/your-username/brain-tumor-classification.git  
   cd brain-tumor-classification  
