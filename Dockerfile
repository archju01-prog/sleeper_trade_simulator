FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend

ENV PORT=8060
EXPOSE 8060

CMD ["sh", "-c", "python -m backend.server --host 0.0.0.0 --port ${PORT:-8060}"]
