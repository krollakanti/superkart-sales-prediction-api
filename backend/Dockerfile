
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all backend files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 7860

# Start Flask application using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:app"]
