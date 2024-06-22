import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import cv2
import numpy as np
from typing import Tuple
from PIL import Image

directory = "./screenshots"
output_directory = "./results"


def remove_green_screen(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]

    th = cv2.threshold(a_channel, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    masked = cv2.bitwise_and(image, image, mask=th)

    m1 = masked.copy()
    m1[th == 0] = (255, 255, 255)

    mlab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)

    dst = cv2.normalize(mlab[:, :, 1], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    threshold_value = 200
    dst_th = cv2.threshold(dst, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]

    mlab[:, :, 1][dst_th == 255] = 127
    img2 = cv2.cvtColor(mlab, cv2.COLOR_LAB2BGR)
    img2_bgra = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
    img2_bgra[th == 0] = (0, 0, 0, 0)

    return img2_bgra


def crop(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        cropped_image = image[y:y + h, x:x + w]

        transparent_background = np.zeros((h, w, 4), dtype=np.uint8)

        transparent_background[:, :, :3] = cropped_image[:, :, :3]
        transparent_background[:, :, 3] = cropped_image[:, :, 3]

        return transparent_background
    else:
        print("Error, no contours")


def crop_to_bounding_box(image: np.ndarray, bounding_box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bounding_box
    return image[y:y + h, x:x + w]


def resize_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def process_image(image_path: str, output_path: str) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to read image from {image_path}")
        return

    image_no_green = remove_green_screen(image)

    cropped_image = crop(image_no_green)
    resized_image = resize_image(cropped_image, scale_factor=0.5)

    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    cv2.imwrite(output_path, resized_image, compression_params)
    img = Image.open(output_path)
    img = img.convert('P', palette=Image.Palette.ADAPTIVE, colors=256)
    img.save(output_path, optimize=True)
    print(f"Processed image saved to {output_path}")


def process_file_by_name(file_name: str):
    input_file_path = Path(directory) / Path(file_name)
    if not input_file_path.is_file():
        print(f"Path {input_file_path} is not a file")
        return

    if input_file_path.suffix not in {".jpg", ".jpeg"}:
        return

    output_file_path = Path(output_directory) / Path(file_name.replace(".jpg", ".png").replace(".jpeg", ".png"))
    process_image(str(input_file_path), str(output_file_path))


if __name__ == "__main__":
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    filenames = list(os.listdir(directory))
    with ProcessPoolExecutor(cpu_count()) as executor:
        executor.map(process_file_by_name, filenames)
