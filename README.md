# ImageClassifierAI
The objective of this project is to develop an image classifier capable of distinguishing between two breeds of cats, namely the Ragdoll and Russian Blue breeds, using images of specimens of these breeds from the Oxford-IIIT Pet Dataset.

# Notebook: Image Classification of Ragdoll and Russian Blue Cats

## Use Case:
The task is to develop an image classifier capable of distinguishing between two cat breeds: Ragdoll and Russian Blue. This classifier will be trained and evaluated using images from the Oxford-IIIT Pet Dataset.

## Objective:
The goal of this notebook is to create an image classification model that can accurately identify whether an image depicts a Ragdoll or a Russian Blue cat. The notebook will handle the entire process from data ingestion to model training and evaluation.

## Steps:

### Data Ingestion:
Downloaded the Oxford-IIIT Pet Dataset and organized the images into 'ragdoll' and 'russianblue' folders.
Uploaded the images to a Google Drive repository for online access.
Mounted the Google Drive in the notebook environment.
Split the data into training and testing sets (80% training, 20% testing) and verified the distribution.

### Preprocessing and Feature Extraction:
Resized all images to a uniform size of 128x128 pixels to standardize input.
Converted images to grayscale to reduce complexity and focus on essential features.
Extracted Histogram of Oriented Gradients (HOG) features to capture the structural information of the images.

### Model Training and Evaluation:
Applied traditional machine learning algorithms: Decision Tree, K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Logistic Regression, and Random Forest.
Used Bagging with Decision Trees and Gradient Boosting with XGBoost to improve performance.
Employed neural networks including a custom-built neural network and an autoencoder for feature extraction followed by classification.
Leveraged Transfer Learning with pre-trained Convolutional Neural Networks (CNN) to achieve the best results.
Evaluated models using accuracy, classification report (precision, recall, F1-score), and confusion matrix.

## Conclusion:
The analysis demonstrated that traditional machine learning algorithms and simpler models generally perform worse than more advanced techniques. Specifically, K-Nearest Neighbors and Decision Trees had lower performance due to challenges with high-dimensional data. Logistic Regression and SVMs showed moderate success but still lagged behind more sophisticated approaches.
Neural networks, particularly the custom-built network and autoencoders, provided better results but still could not match the performance of advanced models. The Bagging Classifier with Decision Trees and Gradient Boosting with XGBoost performed notably well, with XGBoost achieving a high level of accuracy.
The best results were achieved using Transfer Learning with pre-trained CNNs, which utilized models trained on large-scale image datasets (e.g., ImageNet) and provided outstanding performance. This approach confirms that CNNs, especially when pre-trained, are highly effective for complex image classification tasks, outperforming traditional machine learning methods.
