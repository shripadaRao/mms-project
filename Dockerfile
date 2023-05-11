# Use the official Python 3.8.1 image as the base image
FROM python:3.8.1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code to the container
COPY . .

# Set the entry point for the container (replace "app.py" with your program's main file)
CMD ["python", "app.py"]