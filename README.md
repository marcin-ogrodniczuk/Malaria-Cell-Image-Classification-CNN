# Malaria-Cell-Image-Classification-CNN

Overview
I built a deep learning model for malaria cell image classification. The goal was to accurately distinguish between infected and uninfected red blood cell images. To achieve this, I used MobileNetV2 pretrained on ImageNet as the backbone and applied transfer learning with custom dense layers for binary classification. I also implemented rescaling and augmentation to make the model more robust and generalizable.

### Key Tools and Concepts
- TensorFlow / Keras for model building and training
- MobileNetV2 for transfer learning and fine-tuning
- ImageDataGenerator for normalization and augmentation
- Matplotlib / scikit-learn for visualization and evaluation

### Results
- Achieved ~97% accuracy on the test set after fine-tuning
- Strong precision and recall across both classes
- Demonstrated how pretrained CNNs can be adapted for biomedical tasks with limited training data

### Why This Project Matters
Automated malaria detection using deep learning can provide faster and more scalable diagnostic support, especially in resource-limited regions. This project highlights how modern transfer learning techniques can be applied to real-world problems in medical image analysis.

### Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Confirmed model performance on a held-out test batch
