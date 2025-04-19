"""
Cartoon Effect Backend
----------------------
This module handles the image processing operations for the cartoon effect.
It contains functions for each step of the cartoon effect process.

Filters Used:
-------------
1. **Median Filter**: Used for noise reduction by replacing each pixel's value with the median of neighboring pixels.
2. **Laplacian Filter**: Detects edges by calculating the second derivative of the image.
3. **Bilateral Filter**: Smoothens colors while preserving edges to create a cartoonish look.

"""

import cv2
import numpy as np

class CartoonBackend:
    @staticmethod
    def apply_median_filter(image, ksize):
        """
        Apply median filter to reduce noise in the image.
        
        Args:
            image: Input image (BGR format)
            ksize: Kernel size for median filter (must be odd)
            
        Returns:
            Filtered grayscale image
        """
        # Ensure ksize is odd to make sure there is a central pixel
        if ksize % 2 == 0:
            ksize += 1
        
        # Convert to grayscale
        # the inputs are the image and the color conversion code which is cv2.COLOR_BGR2GRAY 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply median filter
        median = cv2.medianBlur(gray, ksize)
        
        return median
    
    @staticmethod
    def detect_edges(image, ksize, threshold):
        """
        Apply Laplacian filter for edge detection and threshold for binary edges.
        
        Args:
            image: Input image (grayscale)
            ksize: Kernel size for Laplacian filter (must be odd)
            threshold: Threshold value for binary conversion
            
        Returns:
            Binary edge mask (white=flat areas, black=edges)
        """
        # Ensure ksize is odd
        if ksize % 2 == 0:
            ksize += 1
        
        # Apply Laplacian filter (second derivative to detect edges)
        edges = cv2.Laplacian(image, cv2.CV_8U, ksize=ksize) ##return type is cv2.CV_8U unsined int of 8 bits
        
        # Apply thresholding: Pixels below 'threshold' become white (255), above become black (0) (inverted binary thresholding)
        # the Inputs are the image, the threshold value, the max value, and the type of thresholding
        # the type of thresholding is cv2.THRESH_BINARY_INV which means that the pixels below the threshold will be set to the max value
        # and the pixels above the threshold will be set to 0
        # the outputs are the threshold value and the thresholded image
        _, edge_mask = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY_INV)
        
        return edge_mask
    
    @staticmethod
    def apply_bilateral_filter(image, d, sigma_color, sigma_space, iterations):
        """
        Apply bilateral filter for color smoothing while preserving edges.
        
        Args:
            image: Input image (BGR format)
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            iterations: Number of times to apply the filter
            
        Returns:
            Filtered color image
        """
        result = image.copy()
        
        # Apply bilateral filter multiple times for stronger effect
        # the goal of iterating the filter is to make the effect stronger.
        for _ in range(iterations):
            result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)
            
        return result
    
    @staticmethod
    def create_cartoon_effect(color_img, edge_mask, edge_thickness=1):
        """
        Combine edge mask with color image to create cartoon effect.
        
        Args:
            color_img: Color image (bilateral filtered)
            edge_mask: Binary edge mask
            edge_thickness: Thickness of edges
            
        Returns:
            Final cartoon image
        """
        # Thicken edges if needed
        if edge_thickness > 1:
            kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
            edge_mask = cv2.dilate(edge_mask, kernel)
        
        # Convert edge mask to BGR for proper combination
        edge_mask_bgr = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
        
        # Create cartoon by masking
        cartoon = np.zeros_like(color_img)
        cartoon[edge_mask_bgr == 255] = color_img[edge_mask_bgr == 255]
        
        return cartoon
    
    @staticmethod
    def process_image(image, params):
        """
        Apply all processing steps to create cartoon effect.
        
        Args:
            image: Input image (BGR format)
            params: Dictionary with processing parameters
                - median_ksize: Kernel size for median filter
                - laplacian_ksize: Kernel size for Laplacian filter
                - threshold: Threshold value for edge detection
                - bilateral_d: Diameter for bilateral filter
                - bilateral_color: Sigma color for bilateral filter
                - bilateral_space: Sigma space for bilateral filter
                - bilateral_iterations: Number of bilateral filter iterations
                - edge_thickness: Thickness of edges in final cartoon
                
        Returns:
            Dictionary with processed images for each step
        """
        # Validate image
        if image is None:
            raise ValueError("No image provided")
        
        # Extract parameters
        median_ksize = params.get('median_ksize', 7)
        laplacian_ksize = params.get('laplacian_ksize', 5)
        threshold = params.get('threshold', 125)
        bilateral_d = params.get('bilateral_d', 9)
        bilateral_color = params.get('bilateral_color', 100)
        bilateral_space = params.get('bilateral_space', 100)
        bilateral_iterations = params.get('bilateral_iterations', 5)
        edge_thickness = params.get('edge_thickness', 1)
        
        # Step 1a: Apply median filter
        median = CartoonBackend.apply_median_filter(image, median_ksize)
        
        # Step 1b: Detect edges
        edge_mask = CartoonBackend.detect_edges(median, laplacian_ksize, threshold)
        
        # Step 2: Apply bilateral filter
        color_img = CartoonBackend.apply_bilateral_filter(
            image, bilateral_d, bilateral_color, bilateral_space, bilateral_iterations
        )
        
        # Step 3: Create cartoon effect
        cartoon = CartoonBackend.create_cartoon_effect(color_img, edge_mask, edge_thickness)
        
        # Return all processed images
        return {
            'median': median,
            'edge_mask': edge_mask,
            'color_img': color_img,
            'cartoon': cartoon
        }


#Done
