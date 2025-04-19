import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Scale, Button, Label, Frame, HORIZONTAL, IntVar, StringVar, Radiobutton, ttk
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import the backend
from lane_detection_backend import LaneDetector

class LaneDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Road Lane Detection using Hough Transform")
        self.root.geometry("1280x900")
        
        self.detector = LaneDetector()
        self.cap = None
        self.video_playing = False
        self.current_frame = None
        self.image_size = (320, 240)  # Standard size for most images
        self.hough_size = (500, 400)  # Larger size for Hough space visualization
        
        # Variables for parameters
        self.canny_low = IntVar(value=50)
        self.canny_high = IntVar(value=150)
        self.hough_threshold = IntVar(value=50)
        self.source_type = StringVar(value="image")
        
        # Create UI elements
        self.create_ui()
    
    def create_ui(self):
        # Main container with scrolling capability
        main_container = Frame(self.root)
        main_container.pack(fill="both", expand=True)
        
        # Add scrollbar
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Top frame for source selection and controls
        top_frame = Frame(scrollable_frame)
        top_frame.pack(fill="x", padx=10, pady=10)
        
        # Source type selection
        Label(top_frame, text="Select Source Type:", font=("Arial", 12)).pack(side="left", padx=5)
        Radiobutton(top_frame, text="Image", variable=self.source_type, value="image", font=("Arial", 11)).pack(side="left", padx=5)
        Radiobutton(top_frame, text="Video", variable=self.source_type, value="video", font=("Arial", 11)).pack(side="left", padx=5)
        
        # Source selection button
        self.source_btn = Button(top_frame, text="Select Source", command=self.select_source, 
                                font=("Arial", 11), bg="#e0e0e0", padx=10)
        self.source_btn.pack(side="left", padx=10)
        
        # Process button
        self.process_btn = Button(top_frame, text="Process", command=self.process_source, 
                                 font=("Arial", 11), bg="#a0d0a0", padx=10)
        self.process_btn.pack(side="left", padx=10)
        
        # Play/Pause button for video (initially hidden)
        self.play_btn = Button(top_frame, text="Play", command=self.toggle_play, 
                              font=("Arial", 11), bg="#a0a0d0", padx=10)
        
        # Parameter adjustment frame
        param_frame = Frame(scrollable_frame)
        param_frame.pack(fill="x", padx=10, pady=5)
        
        # Add title for parameters section
        param_title = Label(param_frame, text="Adjust Parameters", font=("Arial", 12, "bold"))
        param_title.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        
        # Canny parameters
        Label(param_frame, text="Canny Low Threshold:", font=("Arial", 11)).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        Scale(param_frame, from_=0, to=255, orient=HORIZONTAL, variable=self.canny_low, length=200).grid(row=1, column=1, padx=5, pady=5)
        
        Label(param_frame, text="Canny High Threshold:", font=("Arial", 11)).grid(row=1, column=2, padx=5, pady=5, sticky="w")
        Scale(param_frame, from_=0, to=255, orient=HORIZONTAL, variable=self.canny_high, length=200).grid(row=1, column=3, padx=5, pady=5)
        
        # Hough transform parameters
        Label(param_frame, text="Hough Threshold:", font=("Arial", 11)).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        Scale(param_frame, from_=1, to=200, orient=HORIZONTAL, variable=self.hough_threshold, length=200).grid(row=2, column=1, padx=5, pady=5)
        
        # Status label
        self.status_label = Label(param_frame, text="Status: Ready", font=("Arial", 11), fg="blue")
        self.status_label.grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky="w")
        
        # Title for visualization section
        vis_title = Label(scrollable_frame, text="Lane Detection Process Visualization", font=("Arial", 14, "bold"))
        vis_title.pack(pady=(20, 10), anchor="w", padx=10)
        
        # Create display frames in grid layout for better organization
        self.display_frame = Frame(scrollable_frame)
        self.display_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Row 1: Original and Smoothed
        row1_frame = Frame(self.display_frame)
        row1_frame.pack(fill="x", pady=5)
        
        # Original image
        self.original_frame = self.create_display_panel(row1_frame, "1. Original Image", 0)
        
        # Smoothed image
        self.smoothed_frame = self.create_display_panel(row1_frame, "2. Median Smoothing", 1)
        
        # Row 2: Edge Detection and Masked Edges
        row2_frame = Frame(self.display_frame)
        row2_frame.pack(fill="x", pady=5)
        
        # Edge detection
        self.edge_frame = self.create_display_panel(row2_frame, "3. Canny Edge Detection", 0)
        
        # Masked edges
        self.masked_frame = self.create_display_panel(row2_frame, "4. Region of Interest", 1)
        
        # Row 3: Hough Space (larger) and Final Result
        row3_frame = Frame(self.display_frame)
        row3_frame.pack(fill="x", pady=5)
        
        # Larger Hough space with sinusoidal curves
        self.hough_frame = self.create_display_panel(
            row3_frame, 
            "5. Hough Space with Sinusoidal Curves", 
            0, 
            size=self.hough_size
        )
        
        # Final result
        self.result_frame = self.create_display_panel(row3_frame, "6. Lane Detection Result", 1)
        
        # Add a description section at the bottom
        desc_frame = Frame(scrollable_frame, bd=1, relief="solid")
        desc_frame.pack(fill="x", padx=10, pady=(20, 10))
        
        desc_title = Label(desc_frame, text="Process Description", font=("Arial", 12, "bold"))
        desc_title.pack(anchor="w", padx=10, pady=(10, 5))
        
        desc_text = """
1. Original Image: Input image/video frame
2. Median Smoothing: Reduces noise while preserving edges
3. Canny Edge Detection: Identifies edges in the smoothed image
4. Region of Interest: Masks the image to focus on the road area
5. Hough Space: Visualization of how edge points transform to sinusoidal curves in Hough space
   - Each edge point creates a sinusoidal curve in the Hough space
   - Intersections of these curves indicate potential lines in the image
   - Red dots mark the detected lines (strongest intersections)
6. Final Result: Road lanes detected and drawn on the original image
        """
        
        desc_label = Label(desc_frame, text=desc_text, font=("Arial", 11), justify="left")
        desc_label.pack(anchor="w", padx=10, pady=(5, 10))
    
    def create_display_panel(self, parent, title, column_position, size=None):
        """Helper method to create consistent display panels"""
        if size is None:
            size = self.image_size
            
        frame = Frame(parent, bg="#f0f0f0", bd=2, relief="ridge", width=size[0]+40, height=size[1]+60)
        frame.pack(side="left", padx=10, fill="both", expand=True)
        frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Title with background for better visibility
        title_frame = Frame(frame, bg="#d0d0d0", height=30)
        title_frame.pack(fill="x")
        Label(title_frame, text=title, font=("Arial", 11, "bold"), bg="#d0d0d0").pack(pady=5)
        
        # Canvas for image
        canvas = Label(frame, bg="black")
        canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        return {"frame": frame, "canvas": canvas, "size": size}
    
    def select_source(self):
        """Select image or video file"""
        self.status_label.config(text="Status: Selecting source...")
        
        if self.source_type.get() == "image":
            file_path = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
            
            if file_path:
                self.source_path = file_path
                self.status_label.config(text=f"Status: Image loaded - {os.path.basename(file_path)}")
                # Load and display the image
                self.current_frame = cv2.imread(file_path)
                self.display_image(self.current_frame, self.original_frame["canvas"], self.original_frame["size"])
                # Hide play button when image is selected
                self.play_btn.pack_forget()
            else:
                self.status_label.config(text="Status: Image selection cancelled")
        else:
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
            )
            
            if file_path:
                self.source_path = file_path
                self.status_label.config(text=f"Status: Video loaded - {os.path.basename(file_path)}")
                # Open video
                self.cap = cv2.VideoCapture(file_path)
                # Read first frame
                ret, self.current_frame = self.cap.read()
                if ret:
                    self.display_image(self.current_frame, self.original_frame["canvas"], self.original_frame["size"])
                # Show play button
                self.play_btn.pack(side="left", padx=10)
                self.video_playing = False
                self.play_btn.config(text="Play")
            else:
                self.status_label.config(text="Status: Video selection cancelled")
    
    def process_source(self):
        """Process the current frame"""
        if self.current_frame is None:
            self.status_label.config(text="Status: No image/video selected!")
            return
        
        self.status_label.config(text="Status: Processing...")
        self.root.update()  # Update GUI to show status change
        
        # Get parameter values
        canny_low = self.canny_low.get()
        canny_high = self.canny_high.get()
        hough_threshold = self.hough_threshold.get()
        
        try:
            # Process the frame
            processed = self.detector.process_frame(
                self.current_frame, canny_low, canny_high, hough_threshold
            )
            
            # Display results
            self.display_image(self.current_frame, self.original_frame["canvas"], self.original_frame["size"])
            self.display_image(self.detector.get_smoothed_image(), self.smoothed_frame["canvas"], self.smoothed_frame["size"])
            self.display_image(self.detector.get_edges_image(), self.edge_frame["canvas"], self.edge_frame["size"])
            self.display_image(self.detector.get_masked_edges_image(), self.masked_frame["canvas"], self.masked_frame["size"])
            
            # Display the enhanced Hough space visualization
            self.display_image(self.detector.get_hough_space_image(), self.hough_frame["canvas"], self.hough_frame["size"])
            
            self.display_image(processed, self.result_frame["canvas"], self.result_frame["size"])
            
            self.status_label.config(text="Status: Processing complete")
        except Exception as e:
            self.status_label.config(text=f"Status: Error - {str(e)}")
    
    def toggle_play(self):
        """Toggle video playback"""
        if self.cap is None:
            return
        
        self.video_playing = not self.video_playing
        
        if self.video_playing:
            self.play_btn.config(text="Pause")
            self.status_label.config(text="Status: Playing video...")
            self.play_video()
        else:
            self.play_btn.config(text="Play")
            self.status_label.config(text="Status: Video paused")
    
    def play_video(self):
        """Play the video frame by frame"""
        if not self.video_playing or self.cap is None:
            return
        
        ret, self.current_frame = self.cap.read()
        
        if ret:
            # Process and display the frame
            self.process_source()
            # Schedule the next frame
            self.root.after(30, self.play_video)
        else:
            # Restart video when it reaches the end
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.status_label.config(text="Status: Video restarted")
            self.play_video()
    
    def display_image(self, image, canvas, target_size):
        """Convert OpenCV image to Tkinter format and display it"""
        if image is None:
            return
        
        # Get target dimensions
        target_width, target_height = target_size
        
        # Resize image
        resized = cv2.resize(image, (target_width, target_height))
        
        # Convert to RGB if necessary
        if len(resized.shape) == 3:
            display_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            # For grayscale images, convert to 3-channel
            display_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        # Convert to PhotoImage
        img = Image.fromarray(display_image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        canvas.imgtk = imgtk
        canvas.config(image=imgtk)
    
    def on_close(self):
        """Clean up when closing the application"""
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LaneDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()