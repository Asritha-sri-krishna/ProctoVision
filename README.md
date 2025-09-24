# 🎯 ProctoVision

🚨 A real-time exam malpractice detection system using computer vision and AI.

## Features

- **Real-time Detection**: Uses YOLOv8 to detect persons, mobile phones, laptops, and books
- **Object Tracking**: Maintains consistent object IDs using centroid tracking
- **Zone Calibration**: Allows you to define a specific exam area/seat zone
- **Voice Alerts**: Natural language TTS alerts for detected malpractice
- **Logging**: Automatic snapshot saving and CSV logging of events
- **Head Pose Detection**: Detects when student is looking away (when MediaPipe is available)

## ⚙️ Setup

1. **Activate your virtual environment**:
   ```powershell
   .\exam_env\Scripts\Activate.ps1
   ```

2. **Required packages are already installed**:
   - opencv-python (computer vision)
   - ultralytics (YOLOv8 object detection)
   - numpy (numerical operations)
   - pandas (data logging)
   - pyttsx3 (text-to-speech)
   - Pillow (image processing)

##🚀 Usage

### Basic Usage
```powershell
python exam_monitor_local.py
```

### With Pre-defined Zone
If you know the coordinates of your exam area, you can specify them:
```powershell
python exam_monitor_local.py --seat_zone 100 100 500 400
```

### 🎮 Controls
- **Press 'c'**: Calibrate the exam zone by drawing a rectangle
- **Press 'q'**: Quit the application

## 🔎 How It Works

1. **Initialization**: The system loads the YOLOv8 model and starts the webcam
2. **Zone Calibration**: Draw a rectangle around the exam area (desk/seat)
3. **Real-time Monitoring**:
   - Detects 🎭 persons, 📱 phones, 💻 laptops, 📚 books
   - Tracks objects with unique IDs
   - Monitors if objects are inside/outside the exam zone
   - Checks for multiple persons or prohibited items
   - Detects student absence or looking away behavior

## 🚫 Detected Malpractices

-  👥 **Multiple Persons**: More than one person in exam area
-  📱**Mobile Phone**: Phone detected on desk/in exam zone
-  🚶**Student Absence**: No face detected for extended time
-  👀**Looking Away**: Extended periods of looking away from screen (requires MediaPipe)

## Output Files

The system creates a `local_exam_logs/` directory containing:
- `snapshots/`: Screenshots when malpractice is detected
- `alerts.csv`: Log of all detected events with timestamps

## Configuration

Edit the constants at the top of `exam_monitor_local.py`:
- `MIN_CONF`: Detection confidence threshold (0.35)
- `ALERT_PERSIST_FRAMES`: Frames required before triggering alert (8)
- `LOOK_AWAY_FRAMES`: Frames of looking away before warning (18)
- `ABSENCE_SECONDS`: Seconds without face before absence alert (6)

## Notes

- **MediaPipe**: Currently disabled due to Python 3.13 compatibility. Face detection uses fallback method.
- **Performance**: Uses YOLOv8n (nano) model for speed. Change to YOLOv8s for better accuracy if you have a GPU.
- **Camera**: Ensure your webcam is working and not being used by other applications.

## Troubleshooting

1. **Webcam Issues**: Make sure no other applications are using the camera
2. **Performance**: If running slowly, try reducing the frame rate or using a smaller model
3. **False Positives**: Calibrate the exam zone properly to reduce false detections
4. **Audio Issues**: Ensure your system's text-to-speech is working properly

## Future Improvements

- Add MediaPipe support when Python 3.13 compatibility is available
- Implement face recognition for identity verification
- Add network-based monitoring for multiple students
- Include more sophisticated behavior analysis
