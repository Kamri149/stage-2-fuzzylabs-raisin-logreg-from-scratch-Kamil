FROM python:3.11-slim

WORKDIR /app

# Install only numpy
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + artifact
COPY app/model.py app/server.py models/model.npz ./

EXPOSE 8000
CMD ["python", "server.py"]