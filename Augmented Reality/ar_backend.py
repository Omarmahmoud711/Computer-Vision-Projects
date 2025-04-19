import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

class ARBackend:
    """Backend class that handles all image processing and homography calculations for AR overlay."""
    
    def __init__(self):
        self.book_video_path = ""
        self.book_img_path = ""
        self.ar_source_path = ""
        self.output_path = ""
        self.processed_frames = 0
        self.total_frames = 0
        self.progress_callback = None
        self.cancel_requested = False
        # Tracking parameters
        self.max_error_threshold = 3.0  # Max allowable error for homography
        self.min_inliers = 10  # Minimum number of inliers required for valid homography
        self.reference_update_interval = 30  # Frames between reference image updates
        
    def set_paths(self, book_video: str, book_img: str, ar_source: str, output: str):
        """Set the paths for the input and output files."""
        self.book_video_path = book_video
        self.book_img_path = book_img
        self.ar_source_path = ar_source
        self.output_path = output
    
    def set_progress_callback(self, callback):
        """Set a callback function to report progress."""
        self.progress_callback = callback
    
    def update_progress(self, frame_num: int, total_frames: int):
        """Update processing progress."""
        self.processed_frames = frame_num
        self.total_frames = total_frames
        if self.progress_callback:
            self.progress_callback(frame_num, total_frames)
    
    def request_cancel(self):
        """Request cancellation of the processing."""
        self.cancel_requested = True
    
    def reset_cancel(self):
        """Reset cancellation flag."""
        self.cancel_requested = False
    
    def find_correspondences(self, img1: np.ndarray, img2: np.ndarray, 
                            num_correspondences: int = 100) -> Tuple[List, List, List]:
        """
        Find point correspondences between two images using SIFT and KNN matcher.
        
        Args:
            img1: First image
            img2: Second image
            num_correspondences: Number of correspondences to return
            
        Returns:
            Tuple containing (good_matches, keypoints1, keypoints2)
        """
        # Convert images to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        # Check if enough keypoints were detected
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return [], kp1, kp2
        
        # Initialize BFMatcher with default params
        bf = cv2.BFMatcher()
        
        # Apply KNN matcher
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test to filter good matches
        good_matches = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:  # Stricter ratio test (was 0.75)
                good_matches.append(m)
        
        # Sort matches by distance and take the top ones
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        good_matches = good_matches[:num_correspondences]
        
        return good_matches, kp1, kp2
    
    def draw_matches(self, img1: np.ndarray, kp1, img2: np.ndarray, kp2, 
                     matches, output_img_path: Optional[str] = None) -> np.ndarray:
        """
        Draw matches between two images.
        
        Args:
            img1: First image
            kp1: Keypoints from first image
            img2: Second image
            kp2: Keypoints from second image
            matches: List of matches
            output_img_path: Path to save the output image (optional)
            
        Returns:
            Image with matches drawn
        """
        matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        if output_img_path:
            cv2.imwrite(output_img_path, matched_img)
            
        return matched_img
    
    def compute_homography_with_ransac(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute homography using RANSAC and return inlier mask and error.
        
        Args:
            src_pts: Source points (nx2 array)
            dst_pts: Destination points (nx2 array)
            
        Returns:
            Tuple of (homography matrix, inlier mask, reprojection error)
        """
        # Use RANSAC to find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Calculate reprojection error on inliers
        error = 0
        inlier_count = np.sum(mask)
        
        if inlier_count > 0 and H is not None:
            # Transform source points
            src_pts_homogeneous = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))
            transformed_pts = np.dot(H, src_pts_homogeneous.T).T
            
            # Convert from homogeneous coordinates
            transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:]
            
            # Calculate error only for inliers
            mask_indices = mask.ravel() == 1
            if np.any(mask_indices):
                errors = np.sqrt(np.sum((transformed_pts[mask_indices] - dst_pts[mask_indices])**2, axis=1))
                error = np.mean(errors)
        
        return H, mask, error
    
    def get_book_corners(self, book_img: np.ndarray, homography: np.ndarray) -> np.ndarray:
        """
        Calculate the corners of the book in the video frame using homography.
        
        Args:
            book_img: Image of the book cover
            homography: Homography matrix from book cover to video frame
            
        Returns:
            Array of corner coordinates
        """
        h, w = book_img.shape[:2]
        corners = np.array([
            [0, 0],          # top-left
            [w-1, 0],        # top-right
            [w-1, h-1],      # bottom-right
            [0, h-1]         # bottom-left
        ], dtype=np.float32)
        
        # Apply homography to map corners
        corners_homogeneous = np.hstack((corners, np.ones((4, 1))))
        transformed_corners = np.dot(homography, corners_homogeneous.T).T
        
        # Convert back from homogeneous coordinates
        transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]
        
        return transformed_corners
    
    def is_homography_valid(self, H: np.ndarray, corners: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
        """
        Check if homography produces valid transformation (no extreme distortion).
        
        Args:
            H: Homography matrix
            corners: Transformed corners of the book
            frame_shape: Shape of video frame (height, width)
            
        Returns:
            True if homography is valid, False otherwise
        """
        if H is None:
            return False
            
        # Check if corners are inside the frame with some margin
        h, w = frame_shape[:2]
        margin = 0.5  # Allow corners to be up to 50% outside the frame
        
        for corner in corners:
            x, y = corner
            if x < -w*margin or x > w*(1+margin) or y < -h*margin or y > h*(1+margin):
                return False
        
        # Check for reasonable quad area (not too small)
        # Calculate area using Shoelace formula
        x = corners[:, 0]
        y = corners[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        frame_area = w * h
        
        if area < 0.005 * frame_area:  # Less than 0.5% of frame area
            return False
            
        # Check if aspect ratio is preserved approximately
        # Calculate side lengths
        side_lengths = []
        for i in range(4):
            next_i = (i + 1) % 4
            side_lengths.append(np.linalg.norm(corners[i] - corners[next_i]))
        
        width1, height1, width2, height2 = side_lengths
        aspect_ratio1 = max(width1, width2) / max(height1, height2)
        aspect_ratio2 = min(width1, width2) / min(height1, height2)
        
        # Original aspect ratio
        orig_aspect_ratio = w / h
        
        # Allow some deviation in aspect ratio
        if (aspect_ratio1 < 0.5 * orig_aspect_ratio or 
            aspect_ratio1 > 2.0 * orig_aspect_ratio or
            aspect_ratio2 < 0.5 * orig_aspect_ratio or
            aspect_ratio2 > 2.0 * orig_aspect_ratio):
            return False
            
        return True
    
    def crop_ar_frame(self, ar_frame: np.ndarray, book_corners: np.ndarray) -> np.ndarray:
        """
        Crop AR frame to match the book's aspect ratio.
        
        Args:
            ar_frame: Frame from AR source video
            book_corners: Corners of the book in the video frame
            
        Returns:
            Cropped AR frame
        """
        # Calculate book width and height from corners
        width = max(
            np.linalg.norm(book_corners[1] - book_corners[0]),
            np.linalg.norm(book_corners[2] - book_corners[3])
        )
        height = max(
            np.linalg.norm(book_corners[3] - book_corners[0]),
            np.linalg.norm(book_corners[2] - book_corners[1])
        )
        
        # Get aspect ratio
        book_aspect_ratio = width / height
        
        # Get AR frame dimensions
        ar_height, ar_width = ar_frame.shape[:2]
        ar_aspect_ratio = ar_width / ar_height
        
        if ar_aspect_ratio > book_aspect_ratio:
            # AR frame is wider than book
            new_width = int(ar_height * book_aspect_ratio)
            start_x = (ar_width - new_width) // 2
            cropped_frame = ar_frame[:, start_x:start_x+new_width]
        else:
            # AR frame is taller than book
            new_height = int(ar_width / book_aspect_ratio)
            start_y = (ar_height - new_height) // 2
            cropped_frame = ar_frame[start_y:start_y+new_height, :]
        
        return cropped_frame
    
    def overlay_frame(self, book_frame: np.ndarray, ar_frame: np.ndarray, 
                     book_corners: np.ndarray) -> np.ndarray:
        """
        Overlay AR frame onto book in video frame.
        
        Args:
            book_frame: Frame from book video
            ar_frame: Frame from AR source video (cropped)
            book_corners: Corners of the book in the video frame
            
        Returns:
            Frame with AR overlay
        """
        # Create destination points for perspective transform
        h, w = ar_frame.shape[:2]
        dst_pts = np.array([
            [0, 0],
            [w-1, 0],
            [w-1, h-1],
            [0, h-1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(dst_pts, book_corners.astype(np.float32))
        
        # Create mask for the warped AR frame
        mask = np.ones((h, w), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask, M, (book_frame.shape[1], book_frame.shape[0]))
        
        # Apply Gaussian blur to the mask edges for smoother blending
        warped_mask = cv2.GaussianBlur(warped_mask, (5, 5), 2)
        
        # Warp AR frame
        warped_ar = cv2.warpPerspective(ar_frame, M, (book_frame.shape[1], book_frame.shape[0]))
        
        # Create inverse mask and normalize to 0-1 range
        warped_mask = warped_mask.astype(float) / 255.0
        inv_mask = 1.0 - warped_mask
        
        # Use mask to combine images with alpha blending
        result = book_frame.astype(float) * inv_mask[:, :, np.newaxis] + \
                 warped_ar.astype(float) * warped_mask[:, :, np.newaxis]
        
        return result.astype(np.uint8)
    
    def process_videos(self):
        """
        Process videos to create AR overlay video with robust tracking.
        
        Returns:
            Success status and error message if applicable
        """
        try:
            self.reset_cancel()
            
            # Open videos and reference image
            book_cap = cv2.VideoCapture(self.book_video_path)
            ar_cap = cv2.VideoCapture(self.ar_source_path)
            book_img = cv2.imread(self.book_img_path)
            
            if not book_cap.isOpened() or not ar_cap.isOpened() or book_img is None:
                return False, "Failed to open video files or reference image"
            
            # Get video properties
            book_width = int(book_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            book_height = int(book_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = book_cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(book_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ar_total_frames = int(ar_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (book_width, book_height))
            
            # Read first frame from book video to initialize tracking
            ret, first_book_frame = book_cap.read()
            if not ret:
                return False, "Failed to read first frame from book video"
            
            # Find correspondences between book image and first video frame
            good_matches, kp_book, kp_frame = self.find_correspondences(book_img, first_book_frame, 200)
            
            if len(good_matches) < self.min_inliers:
                return False, f"Not enough good matches found in first frame (found {len(good_matches)}, need {self.min_inliers})"
                
            # Extract matched points
            src_pts = np.float32([kp_book[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
            
            # Compute initial homography with RANSAC
            initial_H, mask, error = self.compute_homography_with_ransac(src_pts, dst_pts)
            
            if initial_H is None:
                return False, "Failed to compute initial homography"
                
            inlier_count = np.sum(mask)
            if inlier_count < self.min_inliers:
                return False, f"Not enough inliers for reliable homography (found {inlier_count}, need {self.min_inliers})"
            
            # Reset video captures
            book_cap.release()
            book_cap = cv2.VideoCapture(self.book_video_path)
            
            # Calculate initial book corners
            initial_corners = self.get_book_corners(book_img, initial_H)
            
            # Create reference frame for continuous tracking
            reference_frame = first_book_frame.copy()
            reference_keypoints = kp_frame
            reference_descriptors = cv2.SIFT_create().compute(reference_frame, reference_keypoints)[1]
            
            # Variables to track stable homography
            prev_H = initial_H.copy()
            last_good_H = initial_H.copy()
            lost_tracking_frames = 0
            frame_count = 0
            reference_update_counter = 0
            
            while True:
                if self.cancel_requested:
                    break
                    
                # Read frames from both videos
                book_ret, book_frame = book_cap.read()
                ar_ret, ar_frame = ar_cap.read()
                
                # If we reach the end of book video, break
                if not book_ret:
                    break
                
                # If we reach the end of AR video, loop back to beginning
                if not ar_ret:
                    ar_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    _, ar_frame = ar_cap.read()
                
                # For the first frame, use the initial homography
                if frame_count == 0:
                    H = initial_H
                    corners = initial_corners
                else:
                    # Find correspondences with reference frame
                    sift = cv2.SIFT_create()
                    kp_curr, des_curr = sift.detectAndCompute(book_frame, None)
                    
                    # If no keypoints found, use last good homography
                    if des_curr is None or len(des_curr) < 4:
                        H = last_good_H
                        lost_tracking_frames += 1
                    else:
                        # Match with reference frame descriptors
                        bf = cv2.BFMatcher()
                        matches = bf.knnMatch(reference_descriptors, des_curr, k=2)
                        
                        # Apply more strict ratio test
                        good_matches = []
                        for m, n in matches:
                            if m.distance < 0.7 * n.distance:
                                good_matches.append(m)
                        
                        # If too few matches, try direct book image to current frame matching
                        if len(good_matches) < self.min_inliers:
                            direct_matches, kp_book_direct, kp_curr_direct = self.find_correspondences(book_img, book_frame, 200)
                            
                            if len(direct_matches) >= self.min_inliers:
                                # Extract matched points
                                src_pts = np.float32([kp_book_direct[m.queryIdx].pt for m in direct_matches]).reshape(-1, 2)
                                dst_pts = np.float32([kp_curr_direct[m.trainIdx].pt for m in direct_matches]).reshape(-1, 2)
                                
                                # Compute homography with RANSAC
                                direct_H, mask, error = self.compute_homography_with_ransac(src_pts, dst_pts)
                                
                                # Check if this homography is valid
                                if direct_H is not None:
                                    direct_corners = self.get_book_corners(book_img, direct_H)
                                    
                                    if self.is_homography_valid(direct_H, direct_corners, book_frame.shape) and error < self.max_error_threshold:
                                        H = direct_H
                                        corners = direct_corners
                                        last_good_H = H.copy()
                                        
                                        # Update reference from direct match
                                        reference_frame = book_frame.copy()
                                        reference_keypoints = kp_curr_direct
                                        reference_descriptors = cv2.SIFT_create().compute(reference_frame, reference_keypoints)[1]
                                        lost_tracking_frames = 0
                                        reference_update_counter = 0
                                    else:
                                        H = last_good_H
                                        lost_tracking_frames += 1
                                else:
                                    H = last_good_H
                                    lost_tracking_frames += 1
                            else:
                                H = last_good_H
                                lost_tracking_frames += 1
                        else:
                            # Extract matched points from reference frame
                            ref_pts = np.float32([reference_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                            curr_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                            
                            # Compute homography between reference and current frame
                            frame_H, mask, error = self.compute_homography_with_ransac(ref_pts, curr_pts)
                            
                            if frame_H is None or np.sum(mask) < self.min_inliers or error > self.max_error_threshold:
                                H = last_good_H
                                lost_tracking_frames += 1
                            else:
                                # Calculate combined homography from book image to current frame
                                H = frame_H @ prev_H
                                corners = self.get_book_corners(book_img, H)
                                
                                # Validate combined homography
                                if self.is_homography_valid(H, corners, book_frame.shape):
                                    last_good_H = H.copy()
                                    lost_tracking_frames = 0
                                else:
                                    H = last_good_H
                                    lost_tracking_frames += 1
                
                # If tracking is lost for too many frames, reset with direct matching
                if lost_tracking_frames > 10:
                    # Try direct matching with book image
                    reset_matches, kp_book_reset, kp_curr_reset = self.find_correspondences(book_img, book_frame, 200)
                    
                    if len(reset_matches) >= self.min_inliers:
                        src_pts = np.float32([kp_book_reset[m.queryIdx].pt for m in reset_matches]).reshape(-1, 2)
                        dst_pts = np.float32([kp_curr_reset[m.trainIdx].pt for m in reset_matches]).reshape(-1, 2)
                        
                        reset_H, mask, error = self.compute_homography_with_ransac(src_pts, dst_pts)
                        
                        if reset_H is not None and np.sum(mask) >= self.min_inliers and error < self.max_error_threshold:
                            reset_corners = self.get_book_corners(book_img, reset_H)
                            
                            if self.is_homography_valid(reset_H, reset_corners, book_frame.shape):
                                H = reset_H
                                corners = reset_corners
                                last_good_H = H.copy()
                                prev_H = H.copy()
                                
                                # Reset reference frame
                                reference_frame = book_frame.copy()
                                reference_keypoints = kp_curr_reset
                                reference_descriptors = cv2.SIFT_create().compute(reference_frame, reference_keypoints)[1]
                                lost_tracking_frames = 0
                                reference_update_counter = 0
                
                # Periodically update reference frame to avoid drift
                reference_update_counter += 1
                if reference_update_counter >= self.reference_update_interval and lost_tracking_frames == 0:
                    # Use current frame as new reference
                    reference_frame = book_frame.copy()
                    sift = cv2.SIFT_create()
                    reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_frame, None)
                    prev_H = H.copy()
                    reference_update_counter = 0
                
                # Get book corners for overlay
                corners = self.get_book_corners(book_img, H)
                
                # Crop AR frame to match book aspect ratio
                cropped_ar_frame = self.crop_ar_frame(ar_frame, corners)
                
                # Overlay cropped AR frame onto book
                result_frame = self.overlay_frame(book_frame, cropped_ar_frame, corners)
                
                # Write result frame to output video
                out.write(result_frame)
                
                # Update progress
                frame_count += 1
                self.update_progress(frame_count, total_frames)
            
            # Release resources
            book_cap.release()
            ar_cap.release()
            out.release()
            
            return True, f"Processing completed successfully. Processed {frame_count} frames."
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return False, f"Error during processing: {str(e)}"
        finally:
            # Ensure resources are released even if an error occurs
            try:
                book_cap.release()
            except:
                pass
            
            try:
                ar_cap.release()
            except:
                pass
            
            try:
                out.release()
            except:
                pass

    def extract_frame(self, video_path: str, frame_index: int = 0) -> np.ndarray:
        """Extract a specific frame from a video for preview."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            cap.release()
            return None
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
            
        return frame
        
    def test_homography(self, book_img_path: str, book_video_path: str) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Test homography calculation between reference image and first video frame.
        
        Returns:
            Tuple of (success, message, visualization image)
        """
        try:
            # Read input images
            book_img = cv2.imread(book_img_path)
            
            if book_img is None:
                return False, "Failed to read reference image", None
                
            # Get first frame from video
            cap = cv2.VideoCapture(book_video_path)
            
            if not cap.isOpened():
                return False, "Failed to open video file", None
                
            ret, first_frame = cap.read()
            cap.release()
            
            if not ret:
                return False, "Failed to read first frame from video", None
                
            # Find correspondences
            good_matches, kp_book, kp_frame = self.find_correspondences(book_img, first_frame, 200)
            
            if len(good_matches) < self.min_inliers:
                return False, f"Not enough good matches found (found {len(good_matches)}, need {self.min_inliers})", None
                
            # Extract matched points
            src_pts = np.float32([kp_book[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
            
            # Compute homography with RANSAC
            H, mask, error = self.compute_homography_with_ransac(src_pts, dst_pts)
            
            inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i] == 1]
            
            # Draw and return visualization
            vis_img = self.draw_matches(book_img, kp_book, first_frame, kp_frame, inlier_matches)
            
            # Check if homography is valid
            if H is not None:
                corners = self.get_book_corners(book_img, H)
                valid = self.is_homography_valid(H, corners, first_frame.shape)
                
                if valid:
                    return True, f"Found {len(inlier_matches)} good matches with error {error:.2f}", vis_img
                else:
                    return False, "Homography produces invalid transformation", vis_img
            else:
                return False, "Failed to compute homography", vis_img
            
        except Exception as e:
            return False, f"Error during homography test: {str(e)}", None