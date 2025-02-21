FROM python:3.11-slim

WORKDIR /app/src
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r ../requirements.txt
COPY . /app/
EXPOSE 8050
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8050", "app:server"]
