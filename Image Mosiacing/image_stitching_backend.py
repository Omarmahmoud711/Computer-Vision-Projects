import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

class ImageStitcher:
    def __init__(self):
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        # Feature matcher using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        # Minimum matches required to consider images as having enough correspondence
        self.min_matches = 10
        
    def get_correspondences(self, img1, img2):
        """Get feature correspondences between two images using SIFT"""
        # Convert to grayscale if images are in color
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Find keypoints and descriptors
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        
        # If not enough keypoints found
        if len(kp1) < self.min_matches or len(kp2) < self.min_matches:
            return None, None, None, None, 0
        
        # Match descriptors
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Filter matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        # Check if we have enough good matches
        if len(good_matches) < self.min_matches:
            return None, None, None, None, len(good_matches)
            
        # Extract coordinates of matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        return kp1, kp2, src_pts, dst_pts, len(good_matches)
    
    def compute_homography(self, src_pts, dst_pts):
        """Compute homography matrix between source and destination points"""
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    
    def warp_image(self, img, H, output_shape):
        """Warp the input image using the homography matrix H"""
        return cv2.warpPerspective(img, H, (output_shape[1], output_shape[0]))
    
    def stitch_images(self, img1, img2):
        """Stitch two images together using homography"""
        # Get correspondences
        kp1, kp2, src_pts, dst_pts, num_matches = self.get_correspondences(img1, img2)
        
        # If not enough matches, return None
        if src_pts is None or dst_pts is None:
            return None, num_matches
        
        # Compute homography
        H = self.compute_homography(src_pts, dst_pts)
        
        # Get result dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Create points for corners of img1
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        # Transform corners of img1
        corners1_transformed = cv2.perspectiveTransform(corners1, H)
        
        # Get the minimum and maximum x, y coordinates
        all_corners = np.concatenate((corners1_transformed, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)), axis=0)
        xmin = np.int32(np.min(all_corners[:, 0, 0]))
        ymin = np.int32(np.min(all_corners[:, 0, 1]))
        xmax = np.int32(np.max(all_corners[:, 0, 0]))
        ymax = np.int32(np.max(all_corners[:, 0, 1]))
        
        # Translation matrix to adjust for negative coordinates
        translation_matrix = np.array([
            [1, 0, -xmin],
            [0, 1, -ymin],
            [0, 0, 1]
        ])
        
        # Combine homography with translation
        H_adjusted = translation_matrix @ H
        
        # Dimensions of the panorama
        output_shape = (ymax - ymin, xmax - xmin)
        
        # Warp img1 to align with img2
        img1_warped = cv2.warpPerspective(img1, H_adjusted, (output_shape[1], output_shape[0]))
        
        # Create an empty panorama
        result = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.uint8)
        
        # Place img2 in the result with appropriate offset
        img2_offset_y = -ymin if ymin < 0 else 0
        img2_offset_x = -xmin if xmin < 0 else 0
        
        # Place img2 in the result
        result[img2_offset_y:img2_offset_y + h2, img2_offset_x:img2_offset_x + w2] = img2
        
        # Create a mask for the warped image to blend only non-black parts
        gray_warped = cv2.cvtColor(img1_warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
        mask_3d = np.stack([mask, mask, mask], axis=2) // 255
        
        # Blend the warped image with the result
        result = result * (1 - mask_3d) + img1_warped * mask_3d
        
        return result.astype(np.uint8), num_matches
    
    def stitch_multiple_images(self, images):
        """Stitch multiple images sequentially"""
        if len(images) < 2:
            return images[0] if len(images) == 1 else None, True
            
        # Start with the first image
        result = images[0]
        success = True
        insufficient_matches = []
        
        for i in range(1, len(images)):
            # Stitch the result so far with the next image
            new_result, num_matches = self.stitch_images(images[i], result)
            
            if new_result is None:
                insufficient_matches.append(i)
                success = False
                continue
                
            result = new_result
            
        return result, success, insufficient_matches