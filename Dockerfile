# Use a lightweight but TensorFlow-compatible image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install OpenCV dependencies (libgl1 replaces libgl1-mesa-glx)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default port
EXPOSE 5000

# Start the Flask app using Gunicorn
CMD ["gunicorn", "app_flask:app", "--bind", "0.0.0.0:$PORT"]
