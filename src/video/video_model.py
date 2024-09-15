import joblib
import dlib
from .face_preprocessing import process_video
import numpy as np



trait_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

trait_model_paths = {trait: f"src/video/saved/{trait}_updated_svr_model.pkl" for trait in trait_names}
rf_model_path = "src/video/saved/rf_updated_model.pkl"

class VideoModel:

    '''
        Initializes video model with pretrained face detector , shape predictor and loads models for trait prediction.
    '''

    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("src/video/saved/shape_predictor_68_face_landmarks.dat")
        self.trait_model = {trait: joblib.load(trait_model_paths[trait]) for trait in trait_names}
        self.rf_model = joblib.load(rf_model_path)
    

    def preprocess_video(self , video_path):
        '''
        Preprocesses the video and validates the facial ROI extracted from video frames.


        Parameters:

            - video_path (str): path to the video file

        Returns:

            - np.array: extracted features from the video

        '''
        try:
            
            x_features = process_video(video_path, self.face_detector, self.shape_predictor)
            if x_features is not None:
                return np.array(x_features)
            else:
                raise ValueError(f"Failed to extract features from video: {video_path}")
        except Exception as e:
            print(f"Error during video preprocessing: {str(e)}")
            return None
    

    def predict_traits(self, x_features):

        '''
        Predicts the big five personality traits from the extracted features.


        Parameters:

            - x_features (np.array): extracted features from the video

        Returns:

            - dict: predicted trait values

        '''
        try:
            if x_features is not None:
                trait_preds = {trait: self.trait_model[trait].predict(x_features.reshape(1, -1))[0] for trait in trait_names}
                return trait_preds
            else:
                raise ValueError("Failed to predict traits from video features")
        except Exception as e:
            print(f"Error during trait prediction: {str(e)}")
            return None
        