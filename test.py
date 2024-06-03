import cv2
import numpy as np

# Paths to the model files
prototxt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'

# Load pre-trained model and configuration file
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # confidence threshold
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Class label for 'person'
                count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.putText(frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

