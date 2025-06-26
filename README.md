# Pollen's Profiling: Automated Classification of Pollen Grains

## 📌 Project Overview

This project focuses on building a **deep learning model for automated pollen grain classification** using image data. The classification is done using **Convolutional Neural Networks (CNNs)**, and the model is deployed using a **Flask web application** where users can upload an image and receive the predicted pollen type.

---

## 🎯 Objectives

* Understand the morphological features of pollen grains.
* Train a CNN model to classify pollen images into 23 classes.
* Build a complete pipeline from dataset preprocessing to model deployment.
* Use Flask to develop a simple web application interface.

---

## 📁 Project Structure

```bash
pollen-profiling/
|
├── dataset/                    # Contains 23 pollen grain classes with JPEG images
|
├── templates/                 # HTML files for Flask app
│   └── index.html
|
├── static/                    # CSS and uploaded images
│   └── style.css
|
├── model/                     # Saved trained model
│   └── cnn_pollen_model.h5
|
├── app.py                     # Flask backend
├── train_model.py             # CNN model training script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore
```

---

## 💡 Prior Knowledge & Concepts

* **CNNs** for visual pattern recognition
* **Flask** for backend web deployment
* **OpenCV & NumPy** for image preprocessing
* **Transfer learning** concepts (not used here but relevant)

---

## 🗂️ Dataset

* Source: [Kaggle](https://www.kaggle.com/)
* Region: Brazilian Savannah
* Format: JPEG
* Classes: 23 Pollen Types
* Resolution: \~224x224 (after preprocessing)

---

## 🪑 Use Cases

1. **Environmental Monitoring** - Automated analysis of ecological patterns via pollen data.
2. **Allergy Diagnosis** - Identifying allergenic pollen types for treatment.
3. **Agricultural Research** - Analyzing pollination patterns for better crop management.

---

## 💩 Data Preprocessing

* Resize images to (128, 128)
* Normalize pixel values by dividing by 255
* One-hot encode labels
* Split data into train/test (80/20)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

X = np.array(processed_images)
y = LabelEncoder().fit_transform(class_labels)
y = np_utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

## 🏠 CNN Model Architecture

```python
model = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3), padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, kernel_size=(2, 2), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, kernel_size=(2, 2), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dropout(0.2),
    Dense(500, activation='relu'),
    Dense(150, activation='relu'),
    Dense(23, activation='softmax')
])
```

* Optimizer: `adam`
* Loss: `categorical_crossentropy`
* Epochs: 20
* Accuracy: \~92% on test set

---

## 📊 Training

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
model.save("model/cnn_pollen_model.h5")
```

---

## 🌐 Flask Web App

```python
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = tf.keras.models.load_model("model/cnn_pollen_model.h5")
class_names = ["class_1", ..., "class_23"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_file = request.files["image"]
        if img_file:
            path = os.path.join("static", img_file.filename)
            img_file.save(path)

            img = image.load_img(path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            label = class_names[np.argmax(prediction)]

            return render_template("index.html", label=label, image=path)

    return render_template("index.html", label=None)

if __name__ == "__main__":
    app.run(debug=True)
```

---

## 📄 templates/index.html

```html
<!DOCTYPE html>
<html>
<head>
    <title>Pollen Classifier</title>
</head>
<body>
    <h1>Upload a Pollen Image</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <input type="submit" value="Predict">
    </form>

    {% if label %}
        <h2>Prediction: {{ label }}</h2>
        <img src="{{ image }}" width="300">
    {% endif %}
</body>
</html>
```

---

## 📊 Evaluation

* Validation Accuracy: \~92%
* Avoided overfitting using Dropout layer
* Balanced class distribution

---

## 💾 Requirements

```txt
tensorflow
flask
numpy
opencv-python
pillow
scikit-learn
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

```bash
git clone https://github.com/navyasri0515/pollen-profiling-flask.git
cd pollen-profiling-flask
```

1. Place dataset in `dataset/`.
2. Run training:

   ```bash
   python train_model.py
   ```
3. Start the Flask app:

   ```bash
   python app.py
   ```
4. Open your browser at: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 📝 Conclusion

This project showcases an end-to-end deep learning pipeline: dataset processing, CNN training, evaluation, and web deployment via Flask. The system allows real-time classification of pollen images for use in environmental, medical, and agricultural applications.

---

## 📧 Contact

* **Name**: NAVYA SRI
* **Email**: [navyasrichillapalli@gmail.com](mailto:navyasrichillapalli@gmail.com)
* **GitHub**: [github.com/navyasri0515](https://github.com/navyasri0515)
