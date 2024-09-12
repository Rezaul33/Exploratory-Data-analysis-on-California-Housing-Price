# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the application code to the container
COPY . /app

# Initialize Streamlit config (optional but recommended)
RUN mkdir -p ~/.streamlit

# Copy the custom Streamlit configuration into the container
COPY .streamlit/config.toml ~/.streamlit/config.toml

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default port Streamlit runs on (8501)
EXPOSE 8501

# Define the command to run the application
CMD ["streamlit", "run", "app.py"]
