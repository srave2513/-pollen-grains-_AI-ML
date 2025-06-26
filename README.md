# 🌼 Pollen's Profiling: Automated Classification Of Pollen Grains

## 🚀 Overview

**Pollen's Profiling** is an innovative deep learning project that automates the classification of pollen grains using image processing and Convolutional Neural Networks (CNNs). This tool aims to support environmental monitoring, healthcare diagnosis, and agricultural research through precise and scalable image-based pollen identification.

---

## 📌 Project Objectives

- Develop an accurate image classification system using CNN for pollen grain images.
- Facilitate real-world applications in ecology, allergy treatment, and agriculture.
- Build a user-friendly Flask web application for model inference.

---

## 🔍 Use Cases

### 1. 🌍 Environmental Monitoring
Automated pollen classification aids researchers in tracking biodiversity, pollen seasonality, and ecological trends efficiently and accurately.

### 2. 🩺 Allergy Diagnosis and Treatment
Healthcare professionals can use this tool to detect specific allergenic pollen types, enabling precise treatment and allergen immunotherapy planning.

### 3. 🌾 Agricultural Research and Crop Management
Agronomists can classify pollen grains from different crops, helping improve pollination strategies, breeding techniques, and crop yields.

---

## 🧠 Technologies & Tools Used

- Python
- Jupyter Notebook
- TensorFlow / Keras
- OpenCV
- Flask
- HTML / CSS
- Matplotlib / Seaborn
- Scikit-learn
- Anaconda
- Visual Studio Code

---

## 🗂️ Project Structure

Pollen_Profiling/
│
├── static/
│ └── styles.css
├── templates/
│ ├── index.html
│ └── result.html
│
├── model/
│ └── pollen_model.h5
│
├── dataset/
│ └── [Pollen Images]
│
├── app.py
├── model_training.ipynb
├── requirements.txt
└── README.md


---

## 📥 Dataset

- **Source**: [Kaggle - Brazilian Savannah Pollen Dataset](https://www.kaggle.com/)
- **Contents**: X high-resolution images categorized into Y pollen grain types.
- **Format**: JPEG, manually annotated by palynology experts.

---

## 🧪 Model Architecture (CNN)

```text
Input -> Conv2D(16) -> MaxPool -> 
       Conv2D(32) -> MaxPool -> 
       Conv2D(64) -> MaxPool -> 
       Conv2D(128) -> MaxPool ->
       Flatten -> Dropout(0.2) ->
       Dense(500) -> Dense(150) -> 
       Dense(23, Softmax)
'''
Input Image Size: 128x128
Activation: ReLU

Output: 23 Classes

Total Params: ~4.2M
---
# 🛠️ Setup Instructions
1️⃣ Clone the Repository

git clone https://github.com/yourusername/Pollen-Profiling.git
cd Pollen-Profiling
2️⃣ Create Environment & Install Dependencies

conda create -n pollen_env python=3.8
conda activate pollen_env
pip install -r requirements.txt
3️⃣ Run the Flask Application

python app.py
Open http://127.0.0.1:5000 in your browser.

📊 Image Preprocessing & Augmentation

def process_img(img, size=(128,128)):
    img = cv2.resize(img, size)
    return img / 255.0
Images normalized to [0, 1]

Label encoded using LabelEncoder and to_categorical

Train/Test Split: 80/20

📈 Training Summary
Epochs: 50

Batch Size: 32

Loss Function: Categorical Crossentropy

Optimizer: Adam

Accuracy Achieved: ~95% on test data

💾 Save and Load Model
python
# Save
model.save('model/pollen_model.h5')

# Load
model = load_model('model/pollen_model.h5')
🌐 Flask Web App
Built with Flask and HTML/CSS.

Upload image via browser and get prediction instantly.

app.py
python

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        img_path = os.path.join('static/uploads', file.filename)
        file.save(img_path)

        img = image.load_img(img_path, target_size=(128,128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        return render_template('result.html', prediction=predicted_class, img_path=img_path)
    return render_template('index.html')
📸 UI Screenshots
Upload Page: Select pollen image

Result Page: Displays predicted pollen class and image

📌 Future Enhancements
Add support for multi-label classification

Integrate Grad-CAM for visual explainability

Build API endpoints for mobile integration

Add real-time webcam classification

🧑‍💻 Author
Your Name
GitHub | LinkedIn

📃 License
This project is licensed under the MIT License.

📢 Acknowledgments
Kaggle contributors for the dataset

TensorFlow/Keras community

OpenCV and Flask documentation

⭐️ Don't forget to Star!
If you like this project, give it a ⭐️ on GitHub!


---

Let me know if you'd like the code files (`app.py`, `model_training.ipynb`, or `HTML`) generated too!
