import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import io

# Simple white balance (you can implement other methods as well)
def simple_white_balance(image):
    avg_b = np.mean(image[:,:,0])
    avg_g = np.mean(image[:,:,1])
    avg_r = np.mean(image[:,:,2])
    avg = (avg_r + avg_g + avg_b) / 3
    scale_b = avg / avg_b
    scale_g = avg / avg_g
    scale_r = avg / avg_r
    image[:,:,0] = np.clip(image[:,:,0] * scale_b * 0.9, 0, 255)  # Slight adjustment to the scaling
    image[:,:,1] = np.clip(image[:,:,1] * scale_g * 1.1, 0, 255)  # Slight adjustment to the scaling
    image[:,:,2] = np.clip(image[:,:,2] * scale_r , 0, 255)
    return image

# Function for Gamma Correction
def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Function for Retinex-based Enhancement
def retinex_enhancement(image):
    image = np.float32(image) + 1.0
    ret_image = simple_white_balance(image)
    return np.uint8(ret_image)

def dehaze(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dark_channel = np.min(image, axis=2)
    haze_removed = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 3, 14)
    return haze_removed

# Function for Color Gain
def color_gain(image, r_gain=1.5, g_gain=1.5, b_gain=1.5):
    image[:,:,0] = np.clip(image[:,:,0] * b_gain, 0, 255)
    image[:,:,1] = np.clip(image[:,:,1] * g_gain, 0, 255)
    image[:,:,2] = np.clip(image[:,:,2] * r_gain, 0, 255)
    return image

# Function for Saturation Adjustment
def adjust_saturation(image, saturation_factor=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation_factor, 0, 255)
    image_saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image_saturated

def process_image(image_path):
    # Load the Original Image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Unable to read the image. Please check the file path or format.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 1: Dehazing and Denoising
    image_dehazed = dehaze(image)

    # Step 2: Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    lab = cv2.cvtColor(image_dehazed, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Step 3: Gamma Correction
    gamma_corrected = gamma_correction(image_clahe, gamma=1.8)

    # Step 4: Retinex Enhancement
    image_retinex = retinex_enhancement(gamma_corrected)

    # Step 5: Bilateral Filtering
    image_bilateral = cv2.bilateralFilter(image_retinex, d=5, sigmaColor=25, sigmaSpace=25)

    # Step 6: Apply Color Gain
    image_with_color_gain = color_gain(image_bilateral, r_gain=1.2, g_gain=1.0, b_gain=1.1)

    # Step 7: Adjust Saturation
    image_with_saturation = adjust_saturation(image_with_color_gain, saturation_factor=1.3)

    # Step 8: Merge Original and Enhanced Image
    alpha = 0.45
    beta = 1 - alpha
    merged_image = cv2.addWeighted(image, alpha, image_with_saturation, beta, 0)

    # Display the results
    # First Tab: Comparison Between Original and Enhanced Image
    plt.figure("Comparison", figsize=(12, 6))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Enhanced Image
    plt.subplot(1, 2, 2)
    plt.imshow(merged_image)
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Second Tab: Final Enhanced Image
    plt.figure("Final Enhanced Image", figsize=(12, 6))
    plt.imshow(merged_image)
    plt.title('Enhanced Image')
    plt.axis('off')
    plt.show()


# Script execution
if __name__ == "__main__":
    image_path = sys.argv[1]
    process_image(image_path)
