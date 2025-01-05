import cv2
import pytesseract
import os
import re
import numpy as np
import csv

def clean_text(text):
    """Clean text by removing unwanted characters such as '?' and spaces."""
    return re.sub(r'[^a-zA-Z0-9]', '', text)

def process_images(folder_path):
    if not os.path.exists(folder_path):
        print("The specified folder does not exist.")
        return
    
    # Create output folder
    base_dir = os.path.dirname(folder_path)  # Get the parent directory of folder_path
    output_folder = os.path.join(base_dir, "Opencv_Output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    files = os.listdir(folder_path)
    print(f"Files in folder: {files}")
    
    recognized_plates = []  # Store all recognized license plate texts
    
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing file: {filename}")
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)

            if image is None:
                print(f"Failed to load image: {filename}")
                continue

            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian Blur
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 15, 5)
            
            # Perform edge detection
            edged = cv2.Canny(binary, 50, 500)
            
            # Extract contours
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter potential license plate regions
            plate_contour = None
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                area = cv2.contourArea(contour)

                # Adjust aspect ratio and area conditions
                if 1.0 < aspect_ratio < 8 and 300 < area < 30000:
                    plate_contour = contour
                    break

            if plate_contour is None:
                print(f"No potential license plate detected in {filename}.")
                continue

            # Extract the license plate region
            x, y, w, h = cv2.boundingRect(plate_contour)
            plate = gray[y:y + h, x:x + w]
            
            # Perform OCR on the license plate
            raw_text = pytesseract.image_to_string(plate, config='--psm 8').strip()
            clean_plate_text = clean_text(raw_text)  # Clean the text
            print(f"File: {filename} - Raw Text: {raw_text} - Cleaned Text: {clean_plate_text}")

            if not clean_plate_text:
                clean_plate_text = "unknown_plate"

            # Draw a rectangle and text on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, clean_plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save the result image with detected plate text as filename
            sanitized_filename = clean_text(clean_plate_text)
            output_path = os.path.join(output_folder, f"{sanitized_filename}.jpg")
            cv2.imwrite(output_path, image)
            print(f"Saved result to {output_path}")

            # Add the cleaned text to the list
            recognized_plates.append(clean_plate_text)

    # Save the recognized plates to a CSV file
    csv_path = os.path.join(base_dir, "opencv_recognized_plates.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Opencv License Plate Text"])  # Write header
        for plate_text in recognized_plates:
            csv_writer.writerow([plate_text])  # Write each plate text
    print(f"Recognized plates saved to {csv_path}")

if __name__ == "__main__":
    folder_path = input("Please enter the path to the folder containing images: ").strip()
    process_images(folder_path)
