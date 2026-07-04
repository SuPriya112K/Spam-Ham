# Use a slim, official Python base image - smaller size, faster builds, fewer vulnerabilities
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements.txt first (not the whole project yet)
# This lets Docker cache the pip install layer - if you change your code but not
# requirements.txt, Docker skips reinstalling packages on the next build, saving time.
COPY requirements.txt .

# Install dependencies
# --no-cache-dir keeps the image smaller by not storing pip's download cache
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords at build time (so the container doesn't need internet access to run)
RUN python -c "import nltk; nltk.download('stopwords')"

# Now copy the rest of the project (source code, models)
COPY src/ ./src/
COPY models/ ./models/

# Set the working directory to src/ so relative paths in api.py work correctly
WORKDIR /app/src

# Expose the port FastAPI/uvicorn will run on
EXPOSE 8000

# Command to run when the container starts
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]