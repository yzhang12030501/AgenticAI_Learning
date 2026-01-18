FROM python:3.11-slim

WORKDIR /opt/defaultsite

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./

CMD ["sleep", "infinity"]
