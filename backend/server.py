# server.py
import asyncio
import websockets
import json
import cv2
import numpy as np
import time
import logging
import os
import traceback
from sklearn.preprocessing import LabelEncoder
import joblib
from keras_facenet import FaceNet
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionServer:
    def __init__(self):
        self.initialize_models()
        self.active_connections = set()  # Track all clients

    def initialize_models(self):
        logger.info("Initializing models...")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.facenet = FaceNet()
        self.model = YOLO("yolov8n-face.pt")
        self.svm_model = joblib.load('svm_face_recognition_model1.pkl')
        embeddings_data = np.load("faces-embeddings_done_2classes.npz")
        self.Y = embeddings_data['arr_0']
        self.labels = embeddings_data['arr_1']
        self.encoder = LabelEncoder()
        self.encoded_labels = self.encoder.fit_transform(self.labels)
        logger.info("Models loaded successfully.")

    def calculate_similarity(self, embedding1, embeddings2):
        try:
            return np.dot(embedding1, embeddings2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embeddings2, axis=1))
        except Exception as e:
            logger.error(f"Error in similarity calc: {e}")
            return np.zeros(len(embeddings2))

    async def process_frame(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=frame, show=False, save=False, verbose=False)
        detected_faces = []

        for result in results:
            for box in result.boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_img = rgb_img[y1:y2, x1:x2]
                    if face_img.size == 0: continue
                    face_img = cv2.resize(face_img, (160, 160))
                    face_img = np.expand_dims(face_img, axis=0)
                    embedding = self.facenet.embeddings(face_img)
                    similarities = self.calculate_similarity(embedding, self.Y).flatten()
                    max_similarity = np.max(similarities)
                    if max_similarity > 0.6:
                        face_idx = self.svm_model.predict(embedding)[0]
                        name = self.encoder.inverse_transform([face_idx])[0]
                    else:
                        name = "Visitor"

                    detected_faces.append({
                        "id": str(len(detected_faces)),
                        "name": name,
                        "confidence": float(max_similarity),
                        "box": [x1, y1, x2 - x1, y2 - y1],
                        "timestamp": int(time.time() * 1000)
                    })
                except Exception as e:
                    logger.error(f"Face processing error: {e}")
                    logger.error(traceback.format_exc())
        return detected_faces

    async def video_stream_loop(self):
        cap = cv2.VideoCapture(0)  # Change this to IP stream URL if needed
        if not cap.isOpened():
            logger.error("Failed to open video source.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed.")
                continue

            faces = await self.process_frame(frame)
            response = {
                "status": {
                    "isDetecting": True,
                    "lastUpdated": int(time.time() * 1000)
                },
                "faces": faces
            }

            if self.active_connections:
                logger.info(f"Broadcasting to {len(self.active_connections)} clients.")
            for ws in list(self.active_connections):
                try:
                    await ws.send(json.dumps(response))
                except Exception as e:
                    logger.error(f"Failed to send to client: {e}")

            await asyncio.sleep(0.03)  # ~30 FPS

    async def handle_connection(self, websocket, path):
        client_ip = websocket.remote_address[0]
        self.active_connections.add(websocket)
        logger.info(f"Client connected: {client_ip}")
        try:
            await websocket.wait_closed()
        finally:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected: {client_ip}")

async def main():
    server = FaceRecognitionServer()
    await asyncio.gather(
        websockets.serve(server.handle_connection, "0.0.0.0", 8000),
        server.video_stream_loop()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shut down by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
