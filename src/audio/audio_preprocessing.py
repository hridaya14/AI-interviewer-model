from moviepy.editor import VideoFileClip
import os
import numpy as np
import librosa
import tensorflow as tf
import joblib


audio_path = 'flagged/audio_file/audio.wav'

observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'fearful', 'surprised']


def mp4_to_wav(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio

    if not os.path.exists('flagged/audio_file'):
        os.makedirs('flagged/audio_file')

    audio.write_audiofile(audio_path)

def extract_feature(data, sr, mfcc=True, chroma=True, mel=True):
    result = np.array([])
    if chroma:
        stft = np.abs(librosa.stft(data))
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma_feature = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma_feature))
    if mel:
        mel_feature = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
        result = np.hstack((result, mel_feature))
    return result

def predict_emotion(file_path):
    # Load the model
    model = tf.keras.models.load_model('src/audio/saved/emotion_recognition_model.h5')
    print("Model loaded successfully")

    # Load the label encoder
    label_encoder = joblib.load('src/audio/saved/label_encoder.pkl')

    # Load and preprocess the audio file
    data, sr = librosa.load(file_path)
    segment_length = 5  # Segment length in seconds
    segment_samples = segment_length * sr
    emotion_counts = {emotion: 0 for emotion in observed_emotions}

    num_segments = len(data) // segment_samples
    remaining_samples = len(data) % segment_samples

    # Process each full segment
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = data[start:end]
        feature = extract_feature(segment, sr)
        feature_processed = np.expand_dims(feature, axis=0)
        feature_processed = np.expand_dims(feature_processed, axis=2)

        # Make prediction
        y_pred = model.predict(feature_processed)
        y_pred = np.argmax(y_pred, axis=1)
        predicted_emotion = label_encoder.inverse_transform(y_pred)[0]
        emotion_counts[predicted_emotion] += 1

    # Process the remaining part if it's significant
    if remaining_samples > 0:
        start = num_segments * segment_samples
        segment = data[start:]
        feature = extract_feature(segment, sr)
        feature_processed = np.expand_dims(feature, axis=0)
        feature_processed = np.expand_dims(feature_processed, axis=2)

        # Make prediction
        y_pred = model.predict(feature_processed)
        y_pred = np.argmax(y_pred, axis=1)
        predicted_emotion = label_encoder.inverse_transform(y_pred)[0]
        emotion_counts[predicted_emotion] += 1

    # Calculate percentage
    total_segments = sum(emotion_counts.values())
    emotion_percentages = {emotion: (count / total_segments)  for emotion, count in emotion_counts.items()}
    
    return emotion_percentages