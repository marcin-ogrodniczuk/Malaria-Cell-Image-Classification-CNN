# Malaria-Cell-Image-Classification-CNN

### Project Description
This project implements a deep-learning pipeline to classify microscope images of red blood cells as Parasitized (malaria-infected) or Uninfected. The solution demonstrates how transfer learning can be applied to biomedical image classification tasks with a relatively small, real-world dataset.

### Dataset & Preprocessing
- Dataset: Cell Images for Malaria Detection (public dataset of individual blood-cell images).
- Directory structure: cell_images/Parasitized and cell_images/Uninfected. Hidden/system files were removed and the folder structure was validated before training.
- Image processing: Images were resized to 128×128, converted to float and rescaled to [0, 1].
- Augmentation: On-the-fly augmentation was applied using ImageDataGenerator to increase robustness: small rotations, width/height shifts, shear, zoom, horizontal flip, and a brightness range to simulate staining variation. An 80/20 train/validation split was used.

### Key Tools and Concepts
- TensorFlow / Keras for model building and training
- MobileNetV2 for transfer learning and fine-tuning
- ImageDataGenerator for normalization and augmentation
- Matplotlib / scikit-learn for visualization and evaluation
  
### Model Architecture & Training
- Backbone: MobileNetV2 pretrained on ImageNet (top layers removed).
- Custom head: GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid) for binary classification.

### Training strategy:
- Phase 1: freeze the pretrained backbone and train only the custom head (low compute, fast convergence).
- Phase 2: unfreeze the last ~20 layers of MobileNetV2 and fine-tune with a small learning rate (e.g., Adam(1e-5)) to adapt high-level features to the malaria domain.
- Loss & metrics: binary_crossentropy with accuracy monitored during training. Typical batch size used: 32; epochs: initial head training (≈5–10), fine-tuning (≈5–10), tuned to compute budget.

### Results & Evaluation
- Performance: After fine-tuning, the model reached approximately 97% accuracy on the held-out test batch.
- Sample evaluation (test batch of 32): precision/recall/F1 ~0.97 for both classes with confusion matrix:

[[14  1]
 [ 0 17]]
- This indicates only one misclassification on that test sample and balanced performance across classes.
- Metrics used: accuracy, precision, recall, F1-score, and confusion matrix. ROC/AUC and precision-recall curves are recommended for further analysis in medical settings.

### Interpretability
- Grad-CAM / saliency maps were used to visualize regions of input images that contributed most to model predictions, helping confirm that the network focuses on parasite-like structures rather than irrelevant background artifacts.

### Why this matters
Accurate automated screening tools can provide faster, more scalable diagnostic support, particularly in resource-limited regions where expert microscopy is scarce. This project shows how pretrained CNNs can be adapted efficiently to biomedical tasks, enabling high performance with modest data and compute through transfer learning and careful preprocessing.

### Next steps / extensions
- Expand test coverage with a larger held-out test set and cross-validation.
- Calibrate probabilities and evaluate clinical utility (e.g., decision thresholds tuned to minimize false negatives).
- Improve robustness to different staining protocols and imaging conditions via domain augmentation or stain-normalization techniques.
- Package and deploy as a cloud app or lightweight container for demonstration.
