# Multi-Class CNN Animal Classifier

### Description:
This repository contains code for a project that classifies images of 15 different animal species using a convolutional neural network (CNN). The project utilizes a dataset comprising 378,413 image files, meticulously categorized into 15 classes. The code is written in Python and employs TensorFlow and Keras libraries for model development and training.

### Dataset:
The dataset consists of a vast collection of images belonging to 15 distinct animal species. It includes original training data, augmented training data, and separate testing data for evaluation purposes.

### Model Architecture:
The CNN model architecture is based on the VGG16 model, a renowned deep convolutional neural network developed by the Visual Geometry Group at the University of Oxford. The model is fine-tuned to accommodate the specific requirements of the animal classification task.
![image](https://github.com/Nikhildsaroj/Multi-Class-CNN-Animal-Classifier-15-Animal-Species-/assets/148480961/4889cdcd-6a2c-4a44-ba02-270e44c71f3d)


### Training:
The model is trained using both original and augmented training data to enhance performance and generalization. Early stopping and learning rate scheduling techniques are employed to optimize training efficiency and prevent overfitting.
![image](https://github.com/Nikhildsaroj/Multi-Class-CNN-Animal-Classifier-15-Animal-Species-/assets/148480961/6cb3f2f3-ceb0-41cf-8b6c-e33068767505)


### Evaluation:
The trained model's performance is evaluated using the testing data, with metrics such as loss and accuracy used to assess its classification capabilities.

### Repository Contents:
- **Jupyter Notebook:** The primary notebook contains Python code for data preprocessing, model architecture definition, training, and evaluation. Detailed explanations and comments are provided to guide users through the code implementation.
- **Dataset Files:** The dataset files are included within the repository or can be sourced separately. Instructions for dataset loading and preprocessing are available within the notebook.

### Instructions:
To explore the Multi-Class CNN Animal Classifier project and delve into the intricacies of model development, clone this repository and run the provided Jupyter Notebook. Follow the detailed instructions within to load and preprocess the dataset, define and train the CNN model, and evaluate its performance.

### Model Performance:
- **Training Loss:** 0.0307
- **Test Accuracy:** 89.86%

