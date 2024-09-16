from.audio_preprocessing import mp4_to_wav , predict_emotion

class Audio_Model:

    def __init__ (self):
        self.audio_path = 'flagged/audio_file/audio.wav'
    

    def generate_audio(self, video_path):

        try:
            mp4_to_wav(video_path)
            return self.audio_path
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def predict_emotion(self, audio_path):
        try:
            return predict_emotion(audio_path)
        except Exception as e:
            print(f"Error: {e}")
            return None
        

    
