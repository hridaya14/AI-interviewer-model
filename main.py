import streamlit as st
import os
from src.video.video_model import VideoModel as vm


st.title("Video Face Processing App")
st.write("Upload an MP4 video to process.")


uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file) 

    if uploaded_file.type == "video/mp4":
        
        video_path = "temp/uploaded_video.mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        with open(video_path, mode="wb") as f:
            f.write(uploaded_file.read())

        st.write("Processing the video...")

        video_model = vm()

        x_features = video_model.preprocess_video(video_path)

        if x_features is not None:
            traits = video_model.predict_traits(x_features)
            st.write("Predicted traits:")
            for trait, value in traits.items():
                st.write(f"{trait}: {value}")
        
        st.write("Video processing complete.")
        st.write("Thank you :)")
    
    else:
        st.error("Please upload an MP4 file.")
