FROM apache/airflow:3.3.2
RUN pip install --no-cache-dir numpy opencv-python-headless pillow