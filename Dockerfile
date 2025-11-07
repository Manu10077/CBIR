# Use a lightweight TensorFlow-compatible image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies needed by OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Render assigns
ENV PORT=5000
EXPOSE $PORT

# Start the Flask app using Gunicorn, dynamically binding to Render's port
CMD exec gunicorn app_flask:app --bind 0.0.0.0:${PORT:-5000}
