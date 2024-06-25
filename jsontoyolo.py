import os
import json
import re
from PIL import Image

# Directory paths
yolo_labels_dir = "testing_labels/yolo"
images_dir = "testing_labels"


text_file_path = "testing_labels/label.txt"
json_file_path = "testing_labels/label1.json"

def txt_to_json(text_file_path,json_file_path):
    
    # Read the entire content of the text file
    with open(text_file_path, 'r') as file:
        content = file.read()

    # Split content into individual JSON objects based on the pattern
    json_objects = re.split(r'}\s*{', content)
    json_objects = [obj.strip() for obj in json_objects]

    # Fix the split JSON objects and ensure proper JSON formatting
    for i in range(len(json_objects)):
        if not json_objects[i].startswith('{'):
            json_objects[i] = '{' + json_objects[i]
        if not json_objects[i].endswith('}'):
            json_objects[i] += '}'

    # Create a JSON array string
    json_array_string = '[' + ','.join(json_objects) + ']'

    # Parse the JSON array string into a Python object
    photos = json.loads(json_array_string)

    # Create the final dictionary with the required structure
    final_data = {"photos": photos}

    # Write the new dictionary to a JSON file
    with open(json_file_path, 'w') as file:
        json.dump(final_data, file, indent=4)

    print("JSON file has been created successfully.")



def convert_to_yolo(gtboxes, img_width, img_height):
    yolo_annotations = []
    for box in gtboxes:
        class_id = 0  # Assuming "person" is class 0
        fbox = box["vbox"]
        
        # Calculate center coordinates, width and height
        x_center = (fbox[0] + fbox[2] / 2) / img_width
        y_center = (fbox[1] + fbox[3] / 2) / img_height
        width = fbox[2] / img_width
        height = fbox[3] / img_height
        
        # Append to the annotations list
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations


txt_to_json(text_file_path,json_file_path)

# Ensure YOLO labels directory exists
os.makedirs(yolo_labels_dir, exist_ok=True)

# Read the single JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Process each image annotation in the JSON file
for item in data["photos"]:
    image_id = item["ID"]
    gtboxes = item["gtboxes"]
    
    # Get the corresponding image file path
    image_file = f"{image_id}.jpg"
    image_path = os.path.join(images_dir, image_file)
    
    if not os.path.exists(image_path):
        print(f"Image file {image_file} not found, skipping...")
        continue
    
    # Read the image to get its dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size
    
    # Convert annotations to YOLO format
    yolo_annotations = convert_to_yolo(gtboxes, img_width, img_height)
    
    # Write YOLO annotations to a file
    yolo_label_file = os.path.join(yolo_labels_dir, f"{image_id}.txt")
    with open(yolo_label_file, 'w') as file:
        file.write("\n".join(yolo_annotations))

    print(f"Converted annotations for image {image_id} to YOLO format.")

print("Conversion completed.")
