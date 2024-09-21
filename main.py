import gradio as gr
import os
from src.video.video_model import VideoModel as vm
from src.audio.audio_model import Audio_Model as am

def process_video(uploaded_file):
    if uploaded_file is not None:
        video_path = uploaded_file

        try:
            # Initialize models
            video_model = vm()
            audio_model = am()
            
            # Run video model
            x_features = video_model.preprocess_video(video_path)
            if x_features is None:
                return "Failed to extract features from the video."
            
            video_predictions = video_model.predict_traits(x_features)
            if video_predictions is None:
                return "Failed to predict traits from video features."
            
            # Run audio model
            audio_path = audio_model.generate_audio(video_path)
            if audio_path is None:
                return "Failed to generate audio from video."
            
            audio_predictions = audio_model.predict_emotion(audio_path)
            if audio_predictions is None:
                return "Failed to predict emotion from audio."

            # Combine results
            result_text = "Predicted Traits:\n"
            for trait, value in video_predictions.get("traits", {}).items():
                result_text += f"{trait}: {value}\n"
            
            result_text += f"\nInterview Readiness Score: {video_predictions.get('interview_readiness', 'N/A')}\n"
            
            result_text += "\nPredicted Emotion:\n"
            result_text += f"{audio_predictions}"
            
            return result_text
        
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    return "Please upload a valid MP4 file."

video_input = gr.Video(label="Upload MP4 Video")
output_text = gr.Textbox(label="Predicted Traits & Interview Readiness")

demo = gr.Interface(
    fn=process_video, 
    inputs=video_input, 
    outputs=output_text, 
    live=True,
)

# Launch the app
if __name__ == "__main__": 
    demo.launch(server_name="0.0.0.0", share=True)
