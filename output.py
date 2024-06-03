import cv2

# Create a background subtractor object
bg_subtractor = cv2.bgSubtractorMOG2()

# Initialize video capture (0 for webcam)
cap = cv2.VideoCapture(0)

# Font for displaying the count
font = cv2.FONT_HERSHEY_SIMPLEX

# Keep track of the number of people
person_count = 0

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Apply background subtraction
  foreground = bg_subtractor.apply(frame)

  # Convert to grayscale for easier processing
  gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

  # Apply thresholding to isolate potential people
  thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

  # Find contours (potential people)
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Update person count based on number of contours
  person_count = len(contours)

  # Draw a rectangle around detected people (optional)
  for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

  # Display the frame with person count
  cv2.putText(frame, f"People: {person_count}", (10, 30), font, 1, (0, 255, 0), 2)
  cv2.imshow("People Counter", frame)

  # Exit if 'q' key is pressed
  if cv2.waitKey(1) == ord('q'):
    break

# Release capture and destroy windows
cap.release()
cv2.destroyAllWindows()

