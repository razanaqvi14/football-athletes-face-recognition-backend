# Use the official Python 3 image from Docker Hub
FROM python:3

# Set environment to unbuffered for real-time logs
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to start the Flask app with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
