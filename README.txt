# 😊 Emotion Recognition System

The **Emotion Recognition System** is a computer vision project that detects human faces and classifies emotions in real-time using deep learning. It combines facial recognition and emotion detection to interpret expressions such as happiness, sadness, anger, surprise, and more — making it suitable for applications in mental health, customer analysis, and smart systems.

---

## 🚀 Key Features

- **👤 Facial Recognition:** Identifies individuals in live video feeds or images using pre-trained models such as OpenCV or deep learning frameworks (TensorFlow / Keras).  
- **😄 Emotion Detection:** Classifies emotions (e.g., happy, sad, angry, neutral) using models trained on datasets like **FER2013**.  
- **⚡ Real-Time Processing:** Processes video frames efficiently for instant recognition and emotion classification.  
- **🖥️ Interactive Interface:** Displays detected faces alongside their predicted emotions, with the potential to log results for further analysis.  

---

## 🧠 Why It’s Impressive

- **Technical Depth:** Demonstrates expertise in computer vision, real-time data processing, and deep learning.  
- **Practical Relevance:** Can be adapted for use in **security systems**, **customer sentiment tracking**, or **mental health monitoring**.  
- **Cross-Disciplinary Application:** Integrates **AI, software engineering, and data science** to solve real-world challenges.  

---

## 🧩 Tech Stack

- **Language:** Python  
- **Frameworks & Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib  
- **Dataset:** FER2013 *(not included due to size — see instructions below)*  
- **Model Format:** `.h5` (trained model) and `.json` (label map)

---

## ⚙️ How to Run the Project

### 1️⃣ Prepare Dataset

Download the **FER2013 dataset** from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) or another source.  
Organize it as follows:

Emotion rec/
├── fer2013/
│ ├── train/
│ │ ├── happy/
│ │ ├── sad/
│ │ ├── angry/
│ │ └── ...
│ └── test/
│ ├── happy/
│ ├── sad/
│ ├── angry/
│ └── ...


> ⚠️ Note: The FER2013 dataset is **not included in this repository** due to size limits.  

---

### 2️⃣ Set Up Environment

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt

### 3️⃣ Train the Model

python train_model.py

This will generate two files in your project directory:

emotion_model.h5 — the trained emotion detection model

labels.json — the mapping of class labels

You can tweak training settings such as batch size and epochs in train_model.py.

💡 Training can be slow on CPU — GPU acceleration with TensorFlow is recommended.

### 4️⃣ Run Real-Time Detection

python detect_emotion.py


A webcam window will open showing live detections.
Press q to quit the program.

