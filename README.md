# \# SIGN LANGUAGE TRANSLATOR
A real-time web application that detects and translates American Sign Language (ASL) letters from live camera input or uploaded images. It provides text and speech output to enhance communication accessibility.



## \## Features
\- Live video capture and ASL recognition
\- Image upload for sign prediction
\- Speech synthesis for audio translation
\- Detection history with confidence scores
\- Customizable voice settings

## 

## \## Technologies Used

\- Python, Flask (Backend API)
\- TensorFlow/Keras (Machine Learning Model)
\- HTML, CSS, JavaScript (Frontend)
\- WebRTC for camera access
\- Web Speech API for text-to-speech


## 

## \##Features
🎥 Real-time Video Detection: Live camera capture with instant sign recognition
📷 Image Upload Prediction: Upload images of hand signs for offline translation
🔊 Speech Synthesis: Integrated text-to-speech with customizable voice settings
📋 Detection History: Scrollable history with timestamps and confidence scores
🎛️ Voice Controls: Adjustable speech rate, voice selection, and auto-speech toggle
📱 Responsive Design: Modern, mobile-friendly interface
♿ Accessibility: Enhanced communication tools for deaf and hard-of-hearing users



## \##Technology Stack
Backend: Python, Flask, TensorFlow/Keras
Frontend: HTML5, CSS3, JavaScript
Machine Learning: CNN model trained on ASL dataset
Computer Vision: OpenCV, MediaPipe for hand landmark detection
Speech: Web Speech API, Pyttsx3
Camera: WebRTC for live video streaming


## \##Project Structure
sign-language-translator/

├── app.py                    # Flask backend server

├── train.py                  # Model training script

├── project.py               # Sign detection with MediaPipe

├── index.html               # Frontend interface

├── model/

│   ├── sign\_language\_model.h5

│   └── labels.json

├── dataset/

│   ├── train/

│   └── test/



## \##Installation
#### 1.)Clone the repository
* git clone https://github.com/314Harsh/sign-language-translator.git
* cd sign-language-translator
#### 2.)Create virtual environment (recommended)
* python -m venv venv
  source venv/bin/activate    # On Windows: venv\\Scripts\\activate
#### 3.)Start the Flask backend
* python app.py
#### 4.)Open the frontend
* Open index.html in a modern web browser
* Or access http://localhost:5000 if serving through Flask



### \## Usage
#### Live Camera Detection
1.)Click "Start Camera" to begin live detection
2.)Position your hand in front of the camera
3.)The app will detect and translate ASL letters in real-time
4.)Toggle auto-speech or manually click "Speak Current"



#### Image Upload
1.) Click "Choose File" in the upload section
2.)Select an image containing an ASL hand sign
3.)Click "Upload \& Predict"
4.)View the predicted letter with confidence percentage



#### Voice Settings
1.)Auto Speech: Toggle automatic speech for detections
2.)Speech Rate: Adjust playback speed (0.5x - 2.0x)
3.)Voice Selection: Choose from available system voices


### \##Model Training
To train your own model:
#### 1.)Prepare dataset
* Organize images in dataset/train/ and dataset/test/
* Create folders for each letter (A, B, C, etc.)
#### 2.)Run training script
* python train.py
#### 3.)Model files will be saved to model/ directory


### \##API Endpoints
* GET / - Service information
* GET /health - Health check
* GET /labels - Available labels
* POST /predict - Image prediction (multipart/form-data)
* POST /predict\_url - Predict from image URL



#### \##Browser Compatibility
-Chrome 60+
-Firefox 55+
-Safari 11+
-Edge 79+
Required Permissions: Camera access for live detection
 

### \##Contributing
1.) Fork the repository
2.)Create a feature branch (git checkout -b feature/amazing-feature)
3.)Commit your changes (git commit -m 'Add amazing feature')
4.)Push to the branch (git push origin feature/amazing-feature)
5.)Open a Pull Request



### \##Future Enhancements
* Support for full ASL words and phrase
* Multi-language support
* Mobile app version
* Real-time collaborative translation
* Advanced gesture recognition

 
### \##Troubleshooting
#### Camera not working?
* Ensure camera permissions are granted
* Check if another application is using the camera
* Try refreshing the browser


#### Model predictions inaccurate?
* Ensure good lighting conditions
* Position hand clearly in frame
* Check if the sign matches training data

#### Upload not working?

* Verify backend server is running
* Check browser console for errors



* Ensure image format is supported (JPG, PN
