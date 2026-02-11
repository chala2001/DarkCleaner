# Desktop Image Enhancement Tool

A Java Swing–based desktop application integrated with Python image processing to enhance low-light and dark images. The application allows users to select images from their local system, process them using advanced image enhancement techniques, and visualize the improved output.

## Project Overview
This project demonstrates cross-language integration between a Java desktop interface and a Python-based image processing pipeline. Users can easily browse their local files, select a dark or low-light image, and enhance it using multiple computer vision techniques executed through a Python script.

The system focuses on improving image brightness, contrast, color balance, and overall visual quality.

## Key Features
- User-friendly Java Swing desktop interface
- File explorer integration to select images from the local PC
- Image enhancement using Python and OpenCV
- Brightness and contrast improvement for low-light images
- Noise reduction and color enhancement
- Side-by-side visualization of original and enhanced images

## Image Enhancement Pipeline
The selected image is processed using the following steps:
1. Dehazing and noise reduction
2. Contrast Limited Adaptive Histogram Equalization (CLAHE)
3. Gamma correction for brightness improvement
4. Retinex-based enhancement for color balancing
5. Bilateral filtering to preserve edges
6. Color gain and saturation adjustment
7. Merging original and enhanced images for natural output

## System Architecture
- Java Swing handles the graphical user interface and file selection
- The selected image path is passed to a Python script
- Python processes the image using OpenCV and NumPy
- Enhanced image results are displayed using Matplotlib

## Tech Stack
- Java (Java Swing)
- Python
- OpenCV
- NumPy
- Matplotlib

## How It Works
1. Launch the Java Swing application
2. Click the insert/select image button
3. Choose a dark or low-light image from your PC
4. The image path is passed to the Python processing script
5. The enhanced image is generated and displayed

## Use Case
This tool is useful for:
- Enhancing low-light photographs
- Learning Java–Python integration
- Understanding image enhancement techniques
- Academic and mini-project demonstrations
