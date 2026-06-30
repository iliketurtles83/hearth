FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN adduser --disabled-password --gecos "" app && chown -R app:app /app
USER app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]