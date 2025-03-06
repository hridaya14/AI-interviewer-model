# AI Interviewer Model

## Introduction

AI Interviewer Model is an interview analysis system that leverages machine learning techniques for evaluating video and audio responses. It uses OpenCV-based preprocessing along with Support Vector Regression (SVR) and Random Forest Regression (RFR) for video analysis. For audio processing, it employs a Convolutional Neural Network (CNN). The model is deployed using Gradio for an interactive UI and Docker for containerization.

## File Structure

Below is the file structure visualization of the project:

```
📂 AI-interviewer-model
│── 📂 src                # Trained models and checkpoints
│── 📂 flagged            # Temp Files
│── 📜 requirements.txt   # Dependencies list
│── 📜 README.md          # Project documentation
│── 📜 main.py            # Entry point of the application
│── 📜 Dockerfile         # Docker container setup
```

## Getting Started

Follow these steps to set up and run the project.

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- OpenCV
- Scikit-learn
- PyTorch
- TensorFlow/Keras (for CNN model)
- NumPy
- Pandas
- Gradio
- Docker (for deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hridaya14/AI-interviewer-model.git
   cd AI-interviewer-model
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

To launch gradio app:
```bash
python main.py
```

To build and run the Docker container:
```bash
docker build -t ai-interviewer .
docker run -p 7860:7860 ai-interviewer
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

