FROM ubuntu:latest

WORKDIR /app
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgsl-dev

COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . /app

# RUN chmod +x start.sh
# CMD ["./start.sh"]