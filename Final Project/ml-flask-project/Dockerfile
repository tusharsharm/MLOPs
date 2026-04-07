FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p ml-pipeline/data ml-pipeline/results ml-pipeline/mlruns

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 5000 5001

CMD ["/start.sh"]
