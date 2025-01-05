import os
import platform
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFont, ImageDraw
from ultralytics import YOLO
import re
import csv

def text(img, text, xy=(0, 0), color=(0, 255, 0), size=20):
    pil = Image.fromarray(img)
    s = platform.system()
    if s == "Darwin":  # macOS
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/PingFang.ttc', size)
    ImageDraw.Draw(pil).text(xy, text, font=font, fill=color)
    return np.asarray(pil)

# Check if a file is an image
def is_image_file(filename):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def clean_license_plate(text):
    # Retain only letters and numbers
    return re.sub(r'[^A-Za-z0-9]', '', text)

# Dynamically get the path
path = input("Please enter the folder path containing images: ").strip()

if not os.path.exists(path):
    print("The specified path does not exist. Please check and try again!")
    exit()

parent_dir = os.path.dirname(path)  # Get the parent directory of the input folder
output_folder = os.path.join(parent_dir, "YOLOv8_output")  # Create the output folder in the parent directory
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output folder: {output_folder}")

model = YOLO('/Users/alexlee/Downloads/image/yolov8/runs/detect/train/weights/best.pt')

# Get image files
image_files = [f for f in os.listdir(path) if is_image_file(f)]
if not image_files:
    print("No image files found in the specified folder!")
    exit()

# List to store all recognized license plate texts
all_recognized_texts = []

for i, file in enumerate(image_files):  # Iterate through all images
    full = os.path.join(path, file)
    print(full)

    img = cv2.imdecode(np.fromfile(full, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"File {file} cannot be read, skipping...")
        continue
    img = np.ascontiguousarray(img[:, :, ::-1])

    results = model.predict(img, save=False)
    boxes = results[0].boxes.xyxy

    recognized_text = ""
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        tmp = cv2.cvtColor(img[y1:y2, x1:x2].copy(), cv2.COLOR_RGB2GRAY)
        license_raw = pytesseract.image_to_string(tmp, lang='eng', config='--psm 8')
        license_cleaned = clean_license_plate(license_raw)
        recognized_text += license_cleaned.strip() + "_"
        text_x = x1
        text_y = max(y1 - 30, 0)
        img = text(img, license_cleaned.strip(), (text_x, text_y), (0, 255, 0), 20)

    # Ensure recognized text is not empty
    if not recognized_text.strip():
        recognized_text = "unrecognized"
    else:
        recognized_text = recognized_text.strip("_")

    # Append recognized text to the list
    all_recognized_texts.append(recognized_text)

    # Save the image to the output folder
    output_path = os.path.join(output_folder, f"{recognized_text}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Image saved to: {output_path}")

    # Display the image
    cv2.imshow(f"Image {i+1}: {file}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Save recognized texts to a CSV file in the same folder as YOLOv8_output
csv_path = os.path.join(parent_dir , "yolov8_recognized_plates.csv")
with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["YOLOv8 License Plate Text"])  # Write header
    for plate_text in sorted(all_recognized_texts):  # Sort texts from top to bottom
        csv_writer.writerow([plate_text])
print(f"Recognized texts saved to CSV: {csv_path}")

cv2.destroyAllWindows()
