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
   - Built model using **EfficientNetV2B2** as the base.
   - Compiled model with Adam optimizer and categorical crossentropy.
   - Trained model for defined epochs and monitored accuracy/loss.
  
## Week 2

Model Performance Visualization: Accuracy & Loss Trends

Evaluate the model performance using classification metrics and confusion matrix.

Deployed with Gradio

## Week 3

Build model using **EfficientNetV2B3** AND USING **DenseNet201**

Compare the 3 model accuracy and plot bar chart

deployed using streamlit

[!screenshot](https://github.com/gopikasabu25/AICTE-INTERNSHIP/blob/main/gradio%20output.png)

## 📊 Dataset

This project uses a dataset from **Kaggle**:

[Trash Type Image Dataset — by Farzad Nekouei](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)

It contains images of various garbage types categorized into folders suitable for image classification tasks.

