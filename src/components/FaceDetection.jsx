import { useEffect, useRef, useState } from 'react';

const FaceDetection = ({ onFaceDetected }) => {
  const videoRef = useRef(null);
  const [ws, setWs] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [lastResponse, setLastResponse] = useState(null);

  useEffect(() => {
    const connectWebSocket = () => {
      try {
        console.log('Connecting to WebSocket at ws://localhost:8000...');
        const websocket = new WebSocket('ws://localhost:8000');

        websocket.onopen = () => {
          console.log('WebSocket connected');
          setIsConnected(true);
          setError(null);
        };

        websocket.onclose = (event) => {
          console.log('WebSocket closed', event);
          setIsConnected(false);
          setError(`Connection closed: ${event.reason || 'Unknown reason'}`);
          // Try to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        websocket.onerror = (error) => {
          console.error('WebSocket error:', error);
          setError('WebSocket connection error');
          setIsConnected(false);
        };

        websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('Received message:', data);
            setLastResponse(data);
            onFaceDetected(data);
          } catch (e) {
            console.error('Error parsing message:', e);
            setError('Error parsing server message');
          }
        };

        setWs(websocket);

        return () => {
          if (websocket.readyState === WebSocket.OPEN) {
            websocket.close();
          }
        };
      } catch (e) {
        console.error('WebSocket setup failed:', e);
        setError('WebSocket setup failed');
        setIsConnected(false);
      }
    };

    connectWebSocket();
  }, [onFaceDetected]);

  useEffect(() => {
    if (!videoRef.current) return;

    const constraints = {
      video: {
        width: { ideal: 640 },  // Adjust resolution for better performance
        height: { ideal: 480 },
        facingMode: "user",     // Front-facing camera
        frameRate: { ideal: 30, max: 60 }
      }
    };

    console.log('Requesting camera access...');
    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        console.log('Camera access granted');
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      })
      .catch(err => {
        console.error("Error accessing webcam:", err);
        setError("Failed to access webcam");
      });

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        console.log('Stopping camera stream');
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="relative">
      <video
        ref={videoRef}
        className="w-full rounded-lg aspect-video bg-gray-900"
        autoPlay
        muted
        playsInline
      />
      
      {error && (
        <div className="absolute top-2 left-2 right-2 bg-red-500 text-white px-3 py-2 rounded-lg text-sm">
          {error}
        </div>
      )}

      {lastResponse && (
        <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
          Last response: {lastResponse.faces?.length || 0} faces detected
        </div>
      )}

      <div className="absolute bottom-2 right-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
        WebSocket status: {isConnected ? 'Connected' : 'Disconnected'}
      </div>
    </div>
  );
};

export default FaceDetection;
