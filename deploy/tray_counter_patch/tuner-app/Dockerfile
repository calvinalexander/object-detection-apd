# Tray Counter Tuner with saved runs: docker build -t tray-tuner . && docker run -p 5180:5180 -v tray-runs:/app/runs tray-tuner
FROM node:22-bookworm-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:22-bookworm-slim
# process_run.py needs Python 3 with OpenCV and NumPy, and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-numpy python3-opencv ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /app/dist ./dist
COPY server ./server
VOLUME /app/runs
EXPOSE 5180
CMD ["node", "server/server.mjs"]
