# Fatmawati 3D twin (live viewer) for Cloud Run.  Build context: the staging directory that deploy.sh assembles
# (traffic_twin/, data/fatmawati/net, data/fatmawati/viewer, requirements-twin.txt).
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app
COPY requirements-twin.txt .
RUN pip install --no-cache-dir -r requirements-twin.txt
COPY traffic_twin ./traffic_twin
COPY data/fatmawati/net ./data/fatmawati/net
COPY data/fatmawati/viewer ./data/fatmawati/viewer
RUN useradd --create-home twin && chown -R twin /app
USER twin
# trips, vehicle types and sensor definitions are generated at start-up into data/fatmawati/
CMD ["sh", "-c", "python -m traffic_twin.fatmawati.webapp3d --host 0.0.0.0 --port ${PORT} --speed 5 --start-hour 7 --hours 3"]
