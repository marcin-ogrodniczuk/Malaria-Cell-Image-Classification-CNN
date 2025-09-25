# Malaria-Cell-Image-Classification-CNN

#Overview
This project focuses on malaria cell image classification using deep learning. The goal was to build a model that can distinguish between infected and uninfected red blood cell images with high accuracy.

Towards this objective, I used MobileNetV2 pretrained on ImageNet as the backbone and applied transfer learning with custom dense layers for binary classification. I also implemented data rescaling and augmentation to make the model more robust and better able to generalize.

#Key tools and concepts:
- TensorFlow / Keras for model building and training
- MobileNetV2 for transfer learning and fine-tuning
- ImageDataGenerator for normalization and augmentation
- Matplotlib / scikit-learn for visualization and evaluation

#Results
- Achieved ~97% accuracy on the test set after fine-tuning
- Strong precision and recall across both classes
- Final model demonstrates that pretrained CNNs can be adapted for biomedical tasks with limited training data

#Why It Matters
Automated malaria detection using deep learning can provide faster and more scalable diagnostic support in healthcare, especially in resource-limited regions. This project shows how modern transfer learning techniques can be applied to real-world problems in medical image analysis.

#Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
- Confirmed model performance on a held-out test batch.
