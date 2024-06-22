## What the script does

The script processes images to remove green screens and crops them to their bounding boxes. Here's a detailed step-by-step explanation of what the script does:

1. For each image:
   1. Read the Image: The script reads the image from the screenshots directory.
   2. Remove Green Screen: Converts the image to LAB color space and removes the green background.
   3. Crop the Image: Crops the image to the bounding box of the largest contour found, effectively removing unnecessary transparent areas.
   4. Resize the Image: Resizes the cropped image by a scale factor of `0.5`.
   5. Save the Image: Saves the processed image to the `results` directory with high compression and optimized palette.
   6. Output: The processed image is saved in the results directory with a .png extension.

This script automates the task of processing multiple images, making it efficient and convenient to handle batches of images.

## Installation Guide

1. Clone the Repository
    - First, clone the repository or download the script to your local machine.
2. Install Python
   - Ensure you have Python installed. You can download it from the official Python website.
3. Set up a Virtual Environment (Optional, but recommended)
    - Navigate to the directory where you downloaded the script.
    - Create a virtual environment:
      ```shell
      python -m venv env
      ```
    - Activate the virtual environment:
      - On Windows:
        ```shell
        .\env\Scripts\activate
        ```
      - On macOS/Linux:
        ```shell
        source env/bin/activate
        ```
4. Install dependencies
    - ```shell
      pip install -r requirements.txt
      ```

## Usage Guide
1. Prepare directories
   - Ensure you have a directory named `screenshots` in the same directory as the script. This directory should contain the images you want to process.
   - Ensure you have an empty directory named `results` where the processed images will be saved.
2. Run the script
   - ```shell
     python main.py
     ```
