<h1>Intelligent Traffic Management System</h1>

An advanced traffic management system using YOLOv8, OpenCV, and Scikit-learn to analyze traffic flow, detect congestion, and provide real-time insights.

<h3>🌟 Project Overview</h3>
This project offers a smart solution for urban traffic control. It processes video feeds to detect, track, and classify vehicles, using the collected data to calculate traffic density and speed. This enables more efficient traffic management, congestion prediction, and dynamic signal control.

<h3>✨ Key Features-></h3>

Real-time Vehicle Detection: High-accuracy vehicle detection and classification using YOLOv8.

Vehicle Counting: Accurately counts vehicles passing through specified zones.

Traffic Flow Analysis: Measures traffic volume and speed to determine the flow rate.

Congestion Detection: Uses machine learning to identify congestion based on vehicle density and speed.

Data Visualization: Overlays bounding boxes, tracking IDs, and counts on the video feed for intuitive monitoring.

Scalable & Modular: Designed for easy extension with new features like license plate recognition.

<h3>🏗️ System Architecture</h3>
Video Input: Captures video from files or live feeds using OpenCV.

Frame Preprocessing: Optimizes each frame for the detection model.

Object Detection: YOLOv8 identifies and locates vehicles within frames.

Object Tracking: Assigns a unique ID to each vehicle to track its movement across frames.

Data Analysis: Calculates vehicle count, speed, and traffic density from tracking data.

ML Inference: Scikit-learn models predict congestion levels from the analyzed data.

Output Visualization: Displays or saves the processed frame with data overlays.

<h3>💻 Tech Stack</h3>
Programming Language: Python

Computer Vision: OpenCV

Object Detection: YOLOv8

Machine Learning: Scikit-learn

Numerical Operations: NumPy

Data Handling: Pandas

<h3>🛠️ Installation</h3>
1. Clone the repository:
   git clone https://github.com/lkshayb/Traffix-Landing-Page
   cd Traffix-Landing-Page

2. Install dependencies:
   npm install

3. Run the development server:
   npm start

<h3>🚀 Usage</h3>
  Open "http://localhost:3000" in your browser to see the app.
  
  Edit the code in /src/components/ to customize sections like Navbar, Challenges, and Technology.
  
  Deploy easily with "Vercel" or any static hosting provider.

<h3>🤝 Contributing</h3>
  1. Fork the project
  
  2. Create your feature branch (git checkout -b feature/new-feature)
  
  3. Commit your changes (git commit -m 'Add some feature')
  
  4. Push to the branch (git push origin feature/new-feature)
  
  5. Open a Pull Request

<img width="1918" height="916" alt="image" src="https://github.com/user-attachments/assets/9e5f2e96-575d-42d6-a3c6-2eefd3107377" />

<img width="1918" height="908" alt="image" src="https://github.com/user-attachments/assets/564534c2-5a23-4885-a3ce-6b80c4fa3f72" />

