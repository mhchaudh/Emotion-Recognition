1) Place your dataset folders:
   /Users/hassan/Downloads/coding/Emotion rec/fer2013/train/...
   /Users/hassan/Downloads/coding/Emotion rec/fer2013/test/...

   Each emotion as a directory with images.

2) Create virtualenv and install:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3) Train:
   python train_model.py
   - This saves emotion_model.h5 and labels.json in the project folder.

4) Run webcam detection:
   python detect_emotion.py
   - Press 'q' to quit.

Notes:
- Training can be slow on CPU. Use GPU-enabled TensorFlow if available.
- If your images are colored, the generator converts to grayscale automatically.
- Adjust BATCH_SIZE / EPOCHS in train_model.py if needed.
