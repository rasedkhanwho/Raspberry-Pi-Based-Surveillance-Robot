# Raspberry-Pi-Based-Surveillance-Robot
This project focuses on the development of a Raspberry  Pi-based surveillance robot, designed to detect motion, recognize faces, and alert users  in real-time.

The robot uses:
- **PIR sensor** for motion detection
- **Camera Module** for capturing live video feed
- **Pre-trained CNN model (dlib)** for face detection and recognition
- **Email alerts** with captured images for unknown faces
- **Motor driver and wheels** for mobility (optional)
- **GPIO control** for additional devices (e.g., LED indicators)

---

## ⚙️ Features
- Motion detection using PIR sensor
- Automatic camera activation upon motion detection
- Face recognition using **pre-trained CNN (dlib)**
- Email alerts with image attachment for unknown faces
- Real-time preview on Raspberry Pi display
- GPIO-controlled LED indicator for authorized/unauthorized faces
- Fully automated security monitoring system

---

## 🛠️ Hardware Components
- Raspberry Pi 4B (or compatible model)
- Raspberry Pi Camera Module
- PIR Motion Sensor
- Ultrasonic Sensor (optional for obstacle detection)
- LED (status indicator)
- Motor driver (if mobility is required)
- DC motors & chassis (if mobility is required)
- Power supply / battery pack

---

## 💻 Software & Libraries
- **Python 3**
- **OpenCV** – Image processing
- **face_recognition** – Face detection & recognition
- **dlib** – Deep learning-based facial embeddings
- **picamera2** – Camera access
- **smtplib** – Email sending
- **GPIO Zero / RPi.GPIO** – GPIO control

---
