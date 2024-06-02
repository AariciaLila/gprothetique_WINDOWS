ARG BASE_IMAGE=python:3.9-slim
FROM $BASE_IMAGE as runtime-environment

# Install project requirements
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -U "pip>=21.2,<23.2"
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# Add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

FROM runtime-environment

# Copy the whole project except what is in .dockerignore
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

# Ensure the data directory is copied correctly
COPY --chown=${KEDRO_UID}:${KEDRO_GID} data /home/kedro_docker/data

# Create the mlruns directory and set permissions
RUN mkdir -p /home/kedro_docker/mlruns && chmod -R 777 /home/kedro_docker/mlruns

EXPOSE 5002

CMD ["python", "index.py"]
