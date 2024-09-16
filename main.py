import gradio as gr
import os
from src.video.video_model import VideoModel as vm

def process_video(uploaded_file):
    if uploaded_file is not None:
        video_path = uploaded_file

       
        video_model = vm()
        x_features = video_model.preprocess_video(video_path)

        if x_features is not None:
           
            predictions = video_model.predict_traits(x_features)
            
            if predictions is not None:
                
                result_text = "Predicted Traits:\n"
                for trait, value in predictions["traits"].items():
                    result_text += f"{trait}: {value}\n"
                
                result_text += f"\nInterview Readiness Score: {predictions['interview_readiness']}\n"
                
                return result_text

        return "Video processing failed."
    return "Please upload a valid MP4 file."


video_input = gr.Video(label="Upload MP4 Video")
output_text = gr.Textbox(label="Predicted Traits & Interview Readiness")
upload_btn = gr.Button("Process Video")


demo = gr.Interface(
    fn=process_video, 
    inputs=video_input, 
    outputs=output_text, 
    live=True,
)

# Launch the app
if __name__ == "__main__":
    demo.launch(share = True)
