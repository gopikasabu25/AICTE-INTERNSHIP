# 🧠 Garbage Classification using Transfer Learning

This project aims to classify garbage into various categories using image data 

## 📅 Week 1 Progress

This week focused on preparing the data pipeline and training the model using transfer learning.

### ✅ Tasks Completed:

1. **Importing Required Libraries**  
   Loaded essential libraries like TensorFlow, Keras, NumPy, and Matplotlib.

2. **Dataset Loading**  
   Mounted Google Drive and loaded image dataset from structured folders.

3. **Data Preprocessing**  
   Resized images, normalized pixel values, and explored sample data.

4. **Data Augmentation**  
   Applied transformations (flip, zoom, rotate) to enrich training data.

5. **Train/Validation Dataset Preparation**  
   Split data using `image_dataset_from_directory`.


6. **Model Building and Training**  
   - Built model using **EfficientNetB0** as the base.
   - Compiled model with Adam optimizer and categorical crossentropy.
   - Trained model for defined epochs and monitored accuracy/loss.

## 📊 Dataset

This project uses a dataset from **Kaggle**:

[Trash Type Image Dataset — by Farzad Nekouei](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)

It contains images of various garbage types categorized into folders suitable for image classification tasks.

