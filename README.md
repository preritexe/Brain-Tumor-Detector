🧠 Brain Tumor Detection System

This project focuses on detecting brain tumors from MRI images using a Convolutional Neural Network (CNN). Along with making predictions, I also integrated Grad-CAM to understand why the model makes a particular decision, which helps in making the system more interpretable.

🚀 Features
🧠 CNN model to classify MRI images (Tumor / No Tumor)
📊 Achieves around 97% accuracy
🎯 Threshold tuning to reduce false negatives (important for medical use cases)
🔥 Grad-CAM visualization to highlight regions the model focuses on
🌐 Streamlit-based web app for easy interaction
⚠️ Basic validation to ensure correct type of input image


🖼️ How It Works
Upload a brain MRI image
The model processes the image and predicts whether a tumor is present
The app displays:
Prediction result
Confidence score
Grad-CAM heatmap showing important regions


🧪 Model Details
Architecture: Convolutional Neural Network (CNN)
Input size: 128 × 128
Activation functions: ReLU, Sigmoid
Loss function: Binary Crossentropy
Optimizer: Adam


📊 Evaluation
The model achieves around 97% accuracy.
While accuracy is important, I focused more on reducing false negatives, since missing a tumor is much more critical than a false alarm.
Evaluation was done using:
Confusion Matrix
Classification Report


🔥 Grad-CAM (Model Explainability)
To better understand model decisions, I used Grad-CAM.
🔴 Red areas → regions the model focused on the most
🔵 Blue areas → less important regions
This helps in verifying whether the model is looking at meaningful parts of the MRI instead of irrelevant features.


💻 Tech Stack
Python
TensorFlow / Keras
OpenCV
NumPy
Streamlit


📂 Project Structure
Brain-Tumor-Detector/
│
├── app.py
├── gradcam.py
├── preprocess.py
├── brain_tumor_model.keras
├── requirements.txt


▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py


🌐 Deployment
The app is deployed using Streamlit Community Cloud, allowing users to test the model in real time.


⚠️ Disclaimer
This project is built for learning and demonstration purposes only and should not be used for actual medical diagnosis.


👨‍💻 Author
Developed by Prerit Tapa
