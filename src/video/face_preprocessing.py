import cv2
import numpy as np
import face_recognition

def preprocess_face(frame, face_detector, shape_predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if len(faces) == 0:
        return None

    face = faces[0]
    landmarks = shape_predictor(gray, face)

    # Get eye coordinates
    left_eye = np.mean([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)], axis=0)
    right_eye = np.mean([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)], axis=0)

    # Align face
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Rotate the image
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_face = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

    # Crop face ROI
    face_roi = aligned_face[max(face.top(), 0):min(face.bottom(), h), max(face.left(), 0):min(face.right(), w)]

    # Check if face_roi is valid
    if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
        return None

    try:
        return cv2.resize(face_roi, (224, 224))  # Resize to a standard size
    except cv2.error:
        return None

def extract_features(frame):
    
    face_encoding = face_recognition.face_encodings(frame)
    if len(face_encoding) > 0:
        return face_encoding[0]
    return None

def process_video(video_path, face_detector, shape_predictor):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            preprocessed_face = preprocess_face(frame, face_detector, shape_predictor)
            if preprocessed_face is not None:
                features = extract_features(preprocessed_face)
                if features is not None:
                    frames.append(features)
        except Exception as e:
            print(f"Error processing frame in video {video_path}: {str(e)}")
            continue
    cap.release()
    return np.mean(frames, axis=0) if frames else None