FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN adduser --disabled-password --gecos "" app && chown -R app:app /app
# Seed the persistent runtime-state dir so an empty named volume (hearth-data,
# mounted at /tmp/hearth) inherits app:app ownership on first mount.
RUN mkdir -p /tmp/hearth && chown -R app:app /tmp/hearth
USER app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]