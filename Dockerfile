FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD gunicorn app_flask:app --bind 0.0.0.0:$PORT
