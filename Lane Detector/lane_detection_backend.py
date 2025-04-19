import cv2
import numpy as np
from matplotlib import pyplot as plt
# The goal of this assignment is to implement a lane detection algorithm for a self-driving car.
# The algorithm should be able to detect lane lines in a video stream and draw them on the video frames.
# The algorithm is able to detect straight lane lines only (no curves).

class LaneDetector:
    def __init__(self):
        self.accumulator = None
        self.processed_image = None
        self.edges = None
        self.masked_edges = None
        self.smoothed_image = None
        self.lines = None
        self.hough_space = None

    def median_smoothing(self, image, kernel_size=5):
        """Apply median smoothing to reduce noise"""
        return cv2.medianBlur(image, kernel_size)
    
    def detect_edges(self, image, low_threshold=50, high_threshold=150):
        """Apply Canny edge detection"""
        #the canny edge detection filter first applies a Gaussian filter to smooth the image and remove noise
        #then it calculates the gradient of the image to find the edges by using the Sobel operator
        #finally, it applies non-maximum suppression to thin the edges and hysteresis thresholding to remove weak edges
        return cv2.Canny(image, low_threshold, high_threshold)
    
    def region_of_interest(self, image):
        """Create a masked image containing only the region of interest"""
        height, width = image.shape
        
        # Define a polygon for the region of interest
        #the region of interest is a triangle that covers the bottom half of the image
        #it's an np.array with the coordinates of the vertices of the triangle (bottom left, bottom right, center of the image)
        # polygon = np.array([
        #     [(0, height), (width, height), (width // 2, height // 2)]
        # ])
        #A better region of interest would be a trapezoid that covers the bottom half of the image
        #and extends to the middle of the image at the top.
        #This way, the algorithm can detect lane lines that are further away from the car.
        polygon = np.array([[
        (0, height),            # Bottom-left corner
        (width, height),        # Bottom-right corner
        (3 * width // 4, height // 2),  # Upper-right (3/4 of width, middle height)
        (width // 4, height // 2)  # Upper-left (1/4 of width, middle height)
        ]])
        
        # Create an empty mask
        mask = np.zeros_like(image)
        
        # Fill the polygon with white
        cv2.fillPoly(mask, polygon, 255)
        
        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image, mask)
        
        return masked_image
    
    def hough_transform(self, edges, rho_resolution=1, theta_resolution=np.pi/180, threshold=50):
        """Apply Hough Transform to detect lines"""
        # Apply Hough Transform
        # the inputs are the edge image, the resolution of rho and theta, and the threshold for line detection
        # the edges image is the result of the Canny edge detection
        # the resolution of rho and theta determine the granularity of the Hough space

        # the threshold is the minimum number of votes a line needs to be detected
        lines = cv2.HoughLines(edges, rho_resolution, theta_resolution, threshold)
        
        # Create a proper visualization of Hough space with sinusoidal curves
        height, width = edges.shape
        max_rho = int(np.sqrt(height**2 + width**2))
        accumulator = np.zeros((2*max_rho, 180), dtype=np.uint8)
        hough_space = np.zeros((2*max_rho, 180, 3), dtype=np.uint8)
        
        # Build the hough space more deliberately to show sinusoids
        # First find edge points
        y_indices, x_indices = np.where(edges > 0)
        edge_points = list(zip(x_indices, y_indices))
        
        # Select a subset of points for clearer visualization (if there are too many)
        if len(edge_points) > 100:
            np.random.seed(42)  # For reproducibility
            subset_indices = np.random.choice(len(edge_points), 100, replace=False)
            edge_points_subset = [edge_points[i] for i in subset_indices]
        else:
            edge_points_subset = edge_points
        
        # Create the base accumulator (for line detection)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                # Convert to indices for our accumulator array
                rho_idx = int(rho + max_rho)
                theta_idx = int(theta * 180 / np.pi)
                # Ensure indices are within bounds
                if 0 <= rho_idx < 2*max_rho and 0 <= theta_idx < 180:
                    accumulator[rho_idx, theta_idx] = 255
        
        # Create the visualization of Hough space with sinusoidal curves
        hough_space = cv2.cvtColor(accumulator, cv2.COLOR_GRAY2BGR)
        
        # Draw sinusoidal curves for selected edge points
        for point in edge_points_subset:
            x, y = point
            # Different colors for different points
            color = (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200))
            
            # Calculate the sinusoidal curve for this point
            for theta_idx in range(0, 180, 2):  # Step by 2 for efficiency
                theta = theta_idx * np.pi / 180
                rho = int(x * np.cos(theta) + y * np.sin(theta))
                rho_idx = rho + max_rho
                
                # Ensure we're within bounds
                if 0 <= rho_idx < 2*max_rho and 0 <= theta_idx < 180:
                    # Draw a small circle for better visibility
                    cv2.circle(hough_space, (theta_idx, rho_idx), 1, color, -1)
        
        # Mark the detected lines with a different color (red)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                rho_idx = int(rho + max_rho)
                theta_idx = int(theta * 180 / np.pi)
                
                if 0 <= rho_idx < 2*max_rho and 0 <= theta_idx < 180:
                    cv2.circle(hough_space, (theta_idx, rho_idx), 5, (0, 0, 255), -1)
        
        self.hough_space = hough_space
        return lines, accumulator
    
    def refine_coordinates(self, lines, original_image):
        """Refine the detected lines using non-maximum suppression"""
        if lines is None:
            return []
        
        refined_lines = []
        # Sort lines by votes (assuming first elements have more votes)
        sorted_lines = sorted(lines, key=lambda x: x[0][0])
        
        # Apply non-maximum suppression
        while len(sorted_lines) > 0:
            # Get the line with highest votes
            current_line = sorted_lines[0]
            refined_lines.append(current_line)
            
            # Remove similar lines
            new_sorted_lines = []
            for line in sorted_lines[1:]:
                rho_diff = abs(line[0][0] - current_line[0][0])
                theta_diff = abs(line[0][1] - current_line[0][1])
                
                # If the line is significantly different, keep it
                if rho_diff > 20 or theta_diff > np.pi/15:
                    new_sorted_lines.append(line)
            
            sorted_lines = new_sorted_lines
        
        return refined_lines
    
    def draw_lines(self, image, lines):
        """Draw the detected lines on the image"""
        result_image = np.copy(image)
        
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                # Extend the line to the entire image
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return result_image
    

    #main method
    #-----------
    def process_frame(self, frame, canny_low=50, canny_high=150, hough_threshold=120):
        """Process a single frame for lane detection"""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Apply median smoothing
        self.smoothed_image = self.median_smoothing(gray)
        
        # Apply Canny edge detection
        self.edges = self.detect_edges(self.smoothed_image, canny_low, canny_high)
        
        # Apply region of interest masking
        self.masked_edges = self.region_of_interest(self.edges)
        
        # Apply Hough Transform
        self.lines, self.accumulator = self.hough_transform(self.masked_edges, threshold=hough_threshold)
        
        # Refine the detected lines
        refined_lines = self.refine_coordinates(self.lines, frame)
        
        # Draw the lines on the original image
        self.processed_image = self.draw_lines(frame, refined_lines)
        
        return self.processed_image
    

    # Gets methods:
    #--------------
    def get_accumulator_image(self):
        """Return the Hough accumulator as a displayable image"""
        if self.accumulator is None:
            return None
        
        # Normalize the accumulator for better visualization
        normalized = cv2.normalize(self.accumulator, None, 0, 255, cv2.NORM_MINMAX)
        return normalized
    
    def get_hough_space_image(self):
        """Return the Hough space visualization with sinusoidal curves"""
        return self.hough_space
    
    def get_edges_image(self):
        """Return the edge detection result"""
        return self.edges
    
    def get_masked_edges_image(self):
        """Return the masked edges image"""
        return self.masked_edges
    
    def get_smoothed_image(self):
        """Return the smoothed image"""
        return self.smoothed_image
    
    def get_final_image(self):
        """Return the final processed image with lane lines"""
        return self.processed_image