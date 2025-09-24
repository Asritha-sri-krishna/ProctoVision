"""
exam_monitor_local.py

Real-time exam malpractice monitor (local webcam).
Features:
- Webcam capture (OpenCV)
- YOLOv8 detection (person, cell phone, laptop, book)
- Centroid tracker (lightweight) to keep object IDs
- Zone calibration (draw seat/table rectangle)
- MediaPipe FaceMesh for head-pose (looking away) detection
- TTS voice alerts (pyttsx3) using natural sentences
- Snapshot saving + CSV logging
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[WARNING] MediaPipe not available. Face detection and head pose detection will be disabled.")
    MEDIAPIPE_AVAILABLE = False
    mp = None
import pyttsx3
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
import argparse

# ---------------- CONFIG ----------------
MODEL_NAME = "yolov8n.pt"   # small & fast; change to yolov8s.pt if GPU present
MIN_CONF = 0.30             # Lower threshold for better phone back-side detection
ALERT_PERSIST_FRAMES = 3    # frames required to persist before alert (FAST DETECTION!)
PHONE_ALERT_FRAMES = 2      # Even faster for phones - critical violation!
LOOK_AWAY_FRAMES = 25       # frames of looking away before malpractice alert (0.8 sec) - more moderate
ABSENCE_SECONDS = 5         # seconds without face before absence alert
COOLDOWN_FRAMES = 30        # frames to wait before same alert can trigger again
SAVE_DIR = Path("local_exam_logs")
SNAP_DIR = SAVE_DIR / "snapshots"
LOG_CSV = SAVE_DIR / "alerts.csv"
SAVE_DIR.mkdir(exist_ok=True)
SNAP_DIR.mkdir(exist_ok=True)

# classes we care about (COCO labels)
INTEREST_CLASSES = {"person", "cell phone", "laptop", "book"}  # phone label often "cell phone"

# ---------------- Utilities ----------------
def speak(text):
    """Speak text using pyttsx3 (blocking)."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("[TTS ERROR]", e)

def save_snapshot(frame, tag):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = SNAP_DIR / f"{tag.replace(' ','_')}_{ts}.jpg"
    cv2.imwrite(str(fname), frame)
    return str(fname)

def log_event(event_text, snapshot_path=""):
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "event": event_text, "snapshot": snapshot_path}
    if LOG_CSV.exists():
        df = pd.read_csv(LOG_CSV)
        new_row = pd.DataFrame([row])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(LOG_CSV, index=False)
    print("[LOG]", row)

# ---------------- Simple Centroid Tracker ----------------
class CentroidTracker:
    def __init__(self, maxDisappeared=30):
        # next unique object ID
        self.nextObjectID = 0
        # objectID -> centroid
        self.objects = OrderedDict()
        # objectID -> bounding box (x1,y1,x2,y2)
        self.bboxes = OrderedDict()
        # objectID -> disappeared frames count
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        self.bboxes[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.bboxes[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        """
        rects: list of bounding boxes [x1,y1,x2,y2]
        returns: dict objectID -> bbox
        """
        if len(rects) == 0:
            # mark disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.bboxes

        # compute centroids for input rects
        inputCentroids = []
        for (x1,y1,x2,y2) in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids.append((cX, cY))

        if len(self.objects) == 0:
            for i, c in enumerate(inputCentroids):
                self.register(c, rects[i])
        else:
            # match existing objects to new centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # distance matrix between objectCentroids and inputCentroids
            D = np.linalg.norm(np.array(objectCentroids)[:, None] - np.array(inputCentroids)[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bboxes[objectID] = rects[col]
                self.disappeared[objectID] = 0
                usedRows.add(row); usedCols.add(col)

            # any unused inputCentroids -> register
            for col in range(len(inputCentroids)):
                if col not in usedCols:
                    self.register(inputCentroids[col], rects[col])

            # any unmatched existing objects -> mark disappeared
            for row in range(len(objectCentroids)):
                if row not in usedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

        return self.bboxes

# ---------------- Head pose / looking away helper ----------------
# Initialize face detection (OpenCV-based fallback when MediaPipe unavailable)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if MEDIAPIPE_AVAILABLE:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
else:
    face_mesh = None

def is_looking_away_mediapipe(landmarks, w, h, threshold=0.35):
    # MediaPipe version - nose tip ~ 1, left eye outer ~33, right eye outer ~263
    if not MEDIAPIPE_AVAILABLE or landmarks is None:
        return False
    try:
        nose = landmarks.landmark[1]
        left = landmarks.landmark[33]
        right = landmarks.landmark[263]
        nx = nose.x * w
        lx = left.x * w
        rx = right.x * w
        eye_dist = abs(rx - lx) + 1e-6
        nose_offset = abs(nx - (lx + rx)/2) / eye_dist
        return nose_offset > threshold
    except Exception:
        return False

def is_looking_away_opencv(frame, face_rect, frame_center_x):
    """
    MODERATE head pose detection including distance monitoring
    
    FOCUS: Only flag EXTREME left/right looking (suspected cheating)
    ALLOW: 
    - Normal keyboard usage (looking down to type)
    - Minor head movements 
    - Small side glances (natural behavior)
    
    TRIGGERS:
    - Looking far left/rightmost (40%+ of screen width)
    - Sustained extreme side-looking for 25+ frames (0.8+ seconds)
    
    Returns tuple: (is_looking_away, is_too_far)
    """
    if face_rect is None or len(face_rect) == 0:
        return False, False
    
    try:
        x, y, w, h = face_rect
        face_center_x = x + w // 2
        frame_height, frame_width = frame.shape[:2]
        
        # DISTANCE ANALYSIS: Check if student is sitting too far
        face_area = w * h
        frame_area = frame_width * frame_height
        face_ratio = face_area / frame_area
        
        # If face is too small, student is sitting too far from camera
        min_face_ratio = 0.08  # Face should be at least 8% of frame for proper monitoring
        is_too_far = face_ratio < min_face_ratio
        
        # DIRECTION ANALYSIS: Face position analysis (focus on significant side-looking)
        face_offset = abs(face_center_x - frame_width // 2)
        
        looking_away_score = 0
        
        # Score 1: Face position (most reliable) - focus ONLY on extreme side-looking
        if face_offset > frame_width * 0.4:  # 40% of frame width - EXTREME side-looking only
            looking_away_score += 5  # Very strong indicator for extreme turns
        elif face_offset > frame_width * 0.32:  # 32% threshold - significant side-looking  
            looking_away_score += 3  # Strong indicator
        elif face_offset > frame_width * 0.25:  # 25% threshold - moderate side-looking
            looking_away_score += 1  # Mild indicator (allow some head movement)
        
        # Score 2: Face aspect ratio (secondary)
        face_aspect_ratio = w / h
        if face_aspect_ratio < 0.65:  # Face too narrow (profile view)
            looking_away_score += 2
        elif face_aspect_ratio < 0.75:  # Slightly narrow
            looking_away_score += 1
            
        # Score 3: Face position vertically (only for looking UP significantly, allow looking down for keyboard)
        face_center_y = y + h // 2
        frame_center_y = frame_height // 2
        
        vertical_offset_up = frame_center_y - face_center_y  # Positive if looking up
        vertical_offset_down = face_center_y - frame_center_y  # Positive if looking down
        
        # Only penalize looking UP significantly (could be looking at ceiling, other screens)
        # Be lenient with looking down (keyboard, writing, etc.)
        if vertical_offset_up > frame_height * 0.25:  # Looking up too much
            looking_away_score += 2
        elif vertical_offset_down > frame_height * 0.4:  # Only extreme looking down
            looking_away_score += 1  # Mild penalty for extreme down
        
        # Decision: Looking away if score >= 5 (stricter threshold - only for extreme movements)
        # Allow normal head movements for keyboard usage, typing, minor turns
        # Only flag when student is clearly looking far left/right or way up
        is_looking_away = looking_away_score >= 5 and not is_too_far
                
        return is_looking_away, is_too_far
        
    except Exception as e:
        # If any error in processing, assume not looking away to avoid false positives
        return False

# ---------------- Main monitor ----------------
def run_monitor(seat_zone=None):
    # load model
    print("Loading YOLO model:", MODEL_NAME)
    model = YOLO(MODEL_NAME)
    print("YOLO model loaded successfully!")

    cap = cv2.VideoCapture(0)
    time.sleep(1.0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    tracker = CentroidTracker(maxDisappeared=30)

    # keep counters for temporal persistence keyed by event name
    counters = {"other_person":0, "phone":0, "look_away":0, "too_far":0, "absence":0, "unknown_face":0}
    # cooldown counters to prevent repeated alerts
    cooldowns = {"other_person":0, "phone":0, "look_away":0, "too_far":0, "absence":0, "unknown_face":0}
    # malpractice counters for display
    malpractice_counts = {"other_person":0, "phone":0, "look_away":0, "too_far":0, "absence":0, "total":0}
    
    last_reg_face_time = time.time()
    registered_face_present = True  # we assume student is present initially
    fps_est = cap.get(cv2.CAP_PROP_FPS) or 20
    frames_skipped = 0

    # For voice: produce nicer NL sentences
    def speak_event(template_key, details=""):
        templates = {
            "other_person": "Malpractice detected: another person has entered the exam area.",
            "phone": "Malpractice detected: mobile phone detected on the desk.",
            "look_away": "Please focus on your exam. Keep your eyes on the screen.",
            "too_far": "Please move closer to the camera. Focus on the screen.",
            "absence": "Malpractice detected: student absent from camera.",
            "unknown_face": "Malpractice detected: unknown person detected in front of camera.",
            "camera_block": "Alert: camera appears to be blocked or lighting is too low."
        }
        text = templates.get(template_key, "Malpractice detected.") 
        if details:
            text = f"{text} {details}"
        # speak in background (blocking here for simplicity)
        print("[SPEAK]", text)
        speak(text)

    print("Press 'c' to calibrate seat zone, 'q' to quit.")
    calibrated_zone = seat_zone  # (x1,y1,x2,y2) in pixel coords or None

    # calibration: let user press 'c' and drag a rectangle
    def calibrate_zone():
        print("Calibration: draw rectangle on the frame and press ENTER. Press 'r' to reset.")
        ret, frame = cap.read()
        if not ret:
            return None
        clone = frame.copy()
        roi = cv2.selectROI("Draw seat/table zone then press ENTER or SPACE", clone, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Draw seat/table zone then press ENTER or SPACE")
        if roi == (0,0,0,0):
            return None
        x,y,wc,hc = map(int, roi)
        return (x, y, x+wc, y+hc)

    # main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig = frame.copy()
        if frame is None:
            print("[ERROR] Cannot read webcam frame.")
            break
        
        H, W = frame.shape[:2]
        
        # Enhanced preprocessing for better phone detection from all angles
        enhanced_frame = frame.copy()
        # Increase contrast and brightness slightly for better edge detection
        enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=1.1, beta=10)

        # show calibration instructions
        if calibrated_zone is None:
            cv2.putText(frame, "Press 'c' to calibrate seat/table zone", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # run detection at controlled rate
        # (we run YOLO every frame here; adjust if CPU heavy)
        results = model(enhanced_frame)[0]
        # parse detections with better filtering
        rects = []          # bounding boxes for tracker
        det_classes = {}    # map bbox tuple -> class name
        
        # First pass: collect all valid detections with strict filtering
        valid_detections = []
        if results.boxes is not None:
            for box in results.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                if cls_name not in INTEREST_CLASSES:
                    continue
                    
                # Special lower threshold for phones (critical to detect from any angle!)
                phone_conf_threshold = 0.30  # Raised slightly to reduce false positives
                regular_conf_threshold = MIN_CONF
                
                if cls_name in ("cell phone", "phone", "mobile phone"):
                    if conf < phone_conf_threshold:
                        continue
                else:
                    if conf < regular_conf_threshold:
                        continue
                        
                xyxy = box.xyxy[0].cpu().numpy()
                x1,y1,x2,y2 = map(int, xyxy)
                
                # Stricter size filtering to reduce false positives
                bbox_area = (x2-x1) * (y2-y1)
                frame_area = W * H
                bbox_ratio = bbox_area / frame_area
                
                # Filter based on size and area ratio
                if cls_name in ("cell phone", "phone", "mobile phone"):
                    min_area = 300  # Slightly larger for phones
                    max_ratio = 0.15  # Phones shouldn't take up more than 15% of frame
                elif cls_name == "person":
                    min_area = 1000  # Persons should be reasonably sized
                    max_ratio = 0.8   # Person can take most of frame
                else:
                    min_area = 800
                    max_ratio = 0.3
                
                if bbox_area < min_area or bbox_ratio > max_ratio:
                    continue
                    
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
                valid_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': cls_name,
                    'conf': conf,
                    'area': bbox_area
                })
        
        # Second pass: Remove overlapping detections (keep highest confidence)
        filtered_detections = []
        for i, det in enumerate(valid_detections):
            x1, y1, x2, y2 = det['bbox']
            is_overlapping = False
            
            for j, other_det in enumerate(valid_detections):
                if i == j:
                    continue
                ox1, oy1, ox2, oy2 = other_det['bbox']
                
                # Calculate intersection over union (IoU)
                ix1, iy1 = max(x1, ox1), max(y1, oy1)
                ix2, iy2 = min(x2, ox2), min(y2, oy2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    union = det['area'] + other_det['area'] - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    # If significant overlap and other has higher confidence, skip this one
                    if iou > 0.3 and other_det['conf'] > det['conf']:
                        is_overlapping = True
                        break
            
            if not is_overlapping:
                filtered_detections.append(det)
        
        # Final pass: build rects and det_classes from filtered detections
        for det in filtered_detections:
            bbox = det['bbox']
            rects.append(bbox)
            det_classes[bbox] = det['class']

        # update tracker with rects
        tracked = tracker.update(rects)

        # count persons inside seat zone and outside
        persons_in_zone = 0
        persons_outside = 0
        phone_in_zone = 0

        # evaluate tracked objects
        for oid, bbox in tracked.items():
            x1,y1,x2,y2 = bbox
            cls = det_classes.get((x1,y1,x2,y2), "object")
            # draw bbox + id with better colors
            if cls == "person":
                color = (0, 255, 0)  # green for person
            elif cls in ("cell phone", "phone", "mobile phone"):
                color = (0, 0, 255)  # red for phone - DANGER!
            else:
                color = (255, 165, 0)  # orange for other objects
                
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{cls}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # check if center inside calibrated zone (if calibrated)
            cx = int((x1+x2)/2); cy = int((y1+y2)/2)
            inside_zone = True
            if calibrated_zone is not None:
                zx1, zy1, zx2, zy2 = calibrated_zone
                inside_zone = (zx1 <= cx <= zx2) and (zy1 <= cy <= zy2)
                # draw zone
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (100,255,100), 2)

            if cls == "person":
                if inside_zone:
                    persons_in_zone += 1
                else:
                    persons_outside += 1
            if cls in ("cell phone", "phone", "mobile phone"):
                if inside_zone:
                    phone_in_zone += 1

        # face detection: use MediaPipe to check presence & head pose (if available)
        face_present = False
        look_away_flag = False
        too_far_flag = False
        detected_face_rect = None
        
        if MEDIAPIPE_AVAILABLE and face_mesh is not None:
            # MediaPipe version (more accurate but requires MediaPipe)
            rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            mp_res = face_mesh.process(rgb)
            if mp_res.multi_face_landmarks:
                face_present = True
                last_reg_face_time = time.time()
                # head pose check
                if is_looking_away_mediapipe(mp_res.multi_face_landmarks[0], W, H, threshold=0.35):
                    look_away_flag = True
        else:
            # OpenCV fallback: detect faces and check head pose + distance
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                face_present = True
                last_reg_face_time = time.time()
                
                # Use the largest face (most likely the main subject)
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                detected_face_rect = largest_face
                
                # Check head pose and distance using enhanced OpenCV method
                frame_center_x = W // 2
                is_away, is_far = is_looking_away_opencv(orig, largest_face, frame_center_x)
                look_away_flag = is_away
                too_far_flag = is_far
                    
                # Draw face rectangle for debugging
                x, y, w, h = largest_face
                face_color = (0, 255, 255) if too_far_flag else (255, 255, 0)  # Cyan if too far, yellow if normal
                cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 2)
                
                # Show face detection status with distance info
                face_area = w * h
                frame_area = W * H
                face_ratio = face_area / frame_area
                cv2.putText(frame, f"Face: {w}x{h} ({face_ratio:.3f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
            else:
                # No face detected - could be absence or looking completely away
                face_present = False

        # update counters (temporal smoothing) with cooldown
        # decrease all cooldowns
        for key in cooldowns:
            if cooldowns[key] > 0:
                cooldowns[key] -= 1

        # other person: if >1 person inside zone or any person outside zone near seat
        if persons_in_zone > 1 or persons_outside > 0:
            counters["other_person"] += 1
        else:
            counters["other_person"] = 0

        # Phone detection - IMMEDIATE ALERT for critical violation!
        if phone_in_zone > 0:
            counters["phone"] += 1
        else:
            counters["phone"] = 0

        if look_away_flag:
            counters["look_away"] += 1
        else:
            counters["look_away"] = 0
            
        # distance check: face too small (student sitting too far)
        if too_far_flag:
            counters["too_far"] += 1
        else:
            counters["too_far"] = 0

        # absence check: no face detected for ABSENCE_SECONDS
        if time.time() - last_reg_face_time > ABSENCE_SECONDS:
            counters["absence"] += 1
        else:
            counters["absence"] = 0

        # generate events if counters exceed thresholds and not in cooldown
        events = []
        if counters["other_person"] >= ALERT_PERSIST_FRAMES and cooldowns["other_person"] == 0:
            events.append(("other_person", "MALPRACTICE ALERT: Another person detected in exam area!"))
            counters["other_person"] = 0
            cooldowns["other_person"] = COOLDOWN_FRAMES
            malpractice_counts["other_person"] += 1
            malpractice_counts["total"] += 1
            
        # CRITICAL: Phone detection - IMMEDIATE ALERT!
        if counters["phone"] >= PHONE_ALERT_FRAMES and cooldowns["phone"] == 0:
            events.append(("phone", "CRITICAL MALPRACTICE: Mobile phone detected! Exam violation!"))
            counters["phone"] = 0
            cooldowns["phone"] = COOLDOWN_FRAMES
            malpractice_counts["phone"] += 1
            malpractice_counts["total"] += 1
            
        if counters["absence"] >= (ABSENCE_SECONDS * (fps_est/ max(1,1))) and cooldowns["absence"] == 0:
            events.append(("absence", "Malpractice detected: student absent from camera."))
            counters["absence"] = 0
            cooldowns["absence"] = COOLDOWN_FRAMES
            malpractice_counts["absence"] += 1
            malpractice_counts["total"] += 1
            
        if counters["look_away"] >= LOOK_AWAY_FRAMES and cooldowns["look_away"] == 0:
            events.append(("look_away", "Please look front! Focus on your exam screen and avoid looking to the sides."))
            counters["look_away"] = 0
            cooldowns["look_away"] = COOLDOWN_FRAMES
            malpractice_counts["look_away"] += 1
            malpractice_counts["total"] += 1
            
        # Distance alert: student sitting too far from camera
        if counters["too_far"] >= LOOK_AWAY_FRAMES and cooldowns["too_far"] == 0:
            events.append(("too_far", "Please move closer to the camera and focus on the screen for proper monitoring."))
            counters["too_far"] = 0
            cooldowns["too_far"] = COOLDOWN_FRAMES
            malpractice_counts["too_far"] += 1
            malpractice_counts["total"] += 1

        # if events exist -> speak, save snapshot, log
        for key, msg in events:
            snap = save_snapshot(orig, key)
            log_event(msg, snap)
            # use natural-sounding sentence (can be extended via NLP templates)
            speak_event = msg  # here could run a small NLP routine to expand; kept simple
            speak(speak_event)

        # Enhanced HUD with malpractice counts
        hud1 = f"Persons:{persons_in_zone} | Phones:{phone_in_zone} | Face:{face_present} | LookAway:{counters['look_away']}/{LOOK_AWAY_FRAMES} | TooFar:{counters['too_far']}/{LOOK_AWAY_FRAMES}"
        hud2 = f"MALPRACTICES: Total:{malpractice_counts['total']} | Phones:{malpractice_counts['phone']} | LookAway:{malpractice_counts['look_away']} | TooFar:{malpractice_counts['too_far']} | Others:{malpractice_counts['other_person']}"
        hud3 = f"Confidence: {MIN_CONF} (Phone:0.25) | Phone:{PHONE_ALERT_FRAMES}f | LookAway:{LOOK_AWAY_FRAMES}f | Other:{ALERT_PERSIST_FRAMES}f"
        
        cv2.putText(frame, hud1, (10, H-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, hud2, (10, H-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, hud3, (10, H-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
        # Show detection status with immediate warnings
        if calibrated_zone is not None:
            if phone_in_zone > 0:
                # IMMEDIATE visual warning for phone detection - CRITICAL!
                cv2.putText(frame, "⚠️ PHONE DETECTED! ⚠️", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
                cv2.rectangle(frame, (5, 5), (W-5, H-5), (0,0,255), 5)  # Red border
            elif too_far_flag:
                # Immediate visual feedback for sitting too far
                if counters["too_far"] >= 5:  # After 5 frames (0.17 sec) show warning
                    cv2.putText(frame, "📏 TOO FAR! MOVE CLOSER! 📏", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,165,0), 3)
                    cv2.rectangle(frame, (5, 5), (W-5, H-5), (255,165,0), 3)  # Orange border
                else:
                    cv2.putText(frame, "📏 DISTANCE MONITORING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            elif look_away_flag:
                # Immediate visual feedback for looking away (before full alert)
                if counters["look_away"] >= 5:  # After 5 frames (0.17 sec) show warning
                    cv2.putText(frame, "⚠️ LOOKING AWAY! ⚠️", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 3)
                    cv2.rectangle(frame, (5, 5), (W-5, H-5), (0,165,255), 3)  # Orange border
                else:
                    cv2.putText(frame, "👁️ HEAD POSE MONITORING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            else:
                cv2.putText(frame, "MONITORING ACTIVE ✓", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # show frame
        cv2.imshow("Exam Monitor - Press c to calibrate, q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            new_zone = calibrate_zone()
            if new_zone:
                calibrated_zone = new_zone
                print("Calibrated zone:", calibrated_zone)

    cap.release()
    cv2.destroyAllWindows()

# ---------------- CLI run ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seat_zone", nargs=4, type=int, help="Optional: supply calibrated zone x1 y1 x2 y2")
    args = parser.parse_args()
    zone = tuple(args.seat_zone) if args.seat_zone else None
    run_monitor(zone)