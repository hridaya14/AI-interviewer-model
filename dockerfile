# Use a slim Python image to keep the image size smaller
FROM python:3.10-slim

# Set environment variables to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies required for your application
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching and avoid unnecessary reinstallation
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port your Gradio app will run on
EXPOSE 7860

# Use a health check to ensure your app is running
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:7860/ || exit 1

# Set the command to run the application
CMD ["python", "main.py"]
