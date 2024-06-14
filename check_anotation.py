import cv2
import os

# Paths to the directories
images_dir = 'datasets/newone/images/train'
labels_dir = 'datasets/newone/labels1/train'

# Get sorted list of image files
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')])

def read_annotations(label_path, img_width, img_height):
    """Reads annotations from a label file."""
    annotations = []
    with open(label_path, 'r') as file:
        for line in file:
            # Assuming annotation format: classname x_center y_center width height
            parts = line.strip().split()
            if len(parts) == 5:
                class_name, x_center, y_center, width, height = parts
                x_center = float(x_center) * img_width
                y_center = float(y_center) * img_height
                width = float(width) * img_width
                height = float(height) * img_height
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                annotations.append((class_name, x_min, y_min, x_max, y_max))
    return annotations

def draw_annotations(img, annotations):
    """Draws annotations on the image."""
    for annotation in annotations:
        class_name, x_min, y_min, x_max, y_max = annotation
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Display images with annotations
index = 0
while index < len(image_files):
    image_file = image_files[index]
    image_path = os.path.join(images_dir, image_file)
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_file)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image {image_path}")
        index += 1
        continue
    
    img_height, img_width = img.shape[:2]
    
    # Read the annotations
    if os.path.exists(label_path):
        annotations = read_annotations(label_path, img_width, img_height)
        draw_annotations(img, annotations)
    
    # Show the image with annotations
    cv2.imshow('Image', img)
    
    # Wait for key press to move to the next image
    key = cv2.waitKey(0)
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('n'):  # Press 'n' to move to the next image
        index += 1
    elif key == ord('p') and index > 0:  # Press 'p' to move to the previous image
        index -= 1

# Clean up
cv2.destroyAllWindows()
