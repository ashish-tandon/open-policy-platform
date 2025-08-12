FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt . || true
RUN [ -f requirements.txt ] && pip install --no-cache-dir -r requirements.txt || true
COPY src/ ./src/
EXPOSE 9022
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "9022"]