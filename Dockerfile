FROM python:3.8.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the files and store in the '/app/vggish' directory
RUN curl -o vggish/vggish_model.ckpt https://storage.googleapis.com/audioset/vggish_model.ckpt && \
    curl -o vggish/vggish_pca_params.npz https://storage.googleapis.com/audioset/vggish_pca_params.npz

# Copy the rest of your application's code to the container
COPY . .

EXPOSE 5000

# Set the entry point for the container (replace "app.py" with your program's main file)
CMD ["python", "app.py"]