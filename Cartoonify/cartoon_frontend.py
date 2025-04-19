"""
Cartoon Effect Frontend
----------------------
This module provides a GUI for the cartoon effect application.
It interacts with the backend to process images and display results.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import sys

# Import the backend
from cartoon_backend import CartoonBackend

class CartoonEffectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cartoon Effect Application")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Variables for image processing
        self.original_image = None
        self.current_image = None
        self.processing_thread = None
        self.is_processing = False
        
        # Initialize the backend
        self.backend = CartoonBackend()
        
        # Parameters for cartoon effect
        self.median_ksize = tk.IntVar(value=9)
        self.laplacian_ksize = tk.IntVar(value=5)
        self.threshold_value = tk.IntVar(value=70)
        self.bilateral_diameter = tk.IntVar(value=5)
        self.bilateral_color = tk.IntVar(value=80)
        self.bilateral_space = tk.IntVar(value=80)
        self.bilateral_iterations = tk.IntVar(value=5)
        self.edge_thickness = tk.IntVar(value=1)
        
        # Create the main UI components
        self.create_menu()
        self.create_main_frame()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Load an image to begin.")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create a style for the app
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("Heading.TLabel", font=("Arial", 12, "bold"))
        
    def create_menu(self):
        menu_bar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Save Result", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        
        menu_bar.add_cascade(label="File", menu=file_menu)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)
    
    def create_main_frame(self):
        # Main content frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Right panel for image display
        self.display_frame = ttk.Frame(main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control sections
        ttk.Label(control_frame, text="CARTOON EFFECT CONTROLS", style="Heading.TLabel").pack(pady=10)
        
        # Step 1: Edge Detection Controls
        edge_frame = ttk.LabelFrame(control_frame, text="Step 1: Edge Detection")
        edge_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(edge_frame, text="Median Filter Size:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Scale(edge_frame, from_=3, to=15, variable=self.median_ksize, 
                  orient=tk.HORIZONTAL, command=lambda s: self.update_label("median_size_label", f"Size: {int(float(s))}")).pack(fill=tk.X, padx=5)
        self.median_size_label = ttk.Label(edge_frame, text=f"Size: {self.median_ksize.get()}")
        self.median_size_label.pack(anchor=tk.W, padx=5)
        
        ttk.Label(edge_frame, text="Laplacian Kernel Size:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Scale(edge_frame, from_=1, to=11, variable=self.laplacian_ksize, 
                  orient=tk.HORIZONTAL, command=lambda s: self.update_label("laplacian_size_label", f"Size: {int(float(s))}")).pack(fill=tk.X, padx=5)
        self.laplacian_size_label = ttk.Label(edge_frame, text=f"Size: {self.laplacian_ksize.get()}")
        self.laplacian_size_label.pack(anchor=tk.W, padx=5)
        
        ttk.Label(edge_frame, text="Threshold Value:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Scale(edge_frame, from_=0, to=255, variable=self.threshold_value, 
                  orient=tk.HORIZONTAL, command=lambda s: self.update_label("threshold_label", f"Value: {int(float(s))}")).pack(fill=tk.X, padx=5)
        self.threshold_label = ttk.Label(edge_frame, text=f"Value: {self.threshold_value.get()}")
        self.threshold_label.pack(anchor=tk.W, padx=5)
        
        # Step 2: Bilateral Filter Controls
        color_frame = ttk.LabelFrame(control_frame, text="Step 2: Bilateral Filtering")
        color_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(color_frame, text="Filter Diameter:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Scale(color_frame, from_=5, to=15, variable=self.bilateral_diameter, 
                  orient=tk.HORIZONTAL, command=lambda s: self.update_label("diameter_label", f"Value: {int(float(s))}")).pack(fill=tk.X, padx=5)
        self.diameter_label = ttk.Label(color_frame, text=f"Value: {self.bilateral_diameter.get()}")
        self.diameter_label.pack(anchor=tk.W, padx=5)
        
        ttk.Label(color_frame, text="Color Strength:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Scale(color_frame, from_=10, to=250, variable=self.bilateral_color, 
                  orient=tk.HORIZONTAL, command=lambda s: self.update_label("color_label", f"Value: {int(float(s))}")).pack(fill=tk.X, padx=5)
        self.color_label = ttk.Label(color_frame, text=f"Value: {self.bilateral_color.get()}")
        self.color_label.pack(anchor=tk.W, padx=5)
        
        ttk.Label(color_frame, text="Space Strength:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Scale(color_frame, from_=10, to=250, variable=self.bilateral_space, 
                  orient=tk.HORIZONTAL, command=lambda s: self.update_label("space_label", f"Value: {int(float(s))}")).pack(fill=tk.X, padx=5)
        self.space_label = ttk.Label(color_frame, text=f"Value: {self.bilateral_space.get()}")
        self.space_label.pack(anchor=tk.W, padx=5)
        
        ttk.Label(color_frame, text="Iterations:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Scale(color_frame, from_=1, to=10, variable=self.bilateral_iterations, 
                  orient=tk.HORIZONTAL, command=lambda s: self.update_label("iterations_label", f"Value: {int(float(s))}")).pack(fill=tk.X, padx=5)
        self.iterations_label = ttk.Label(color_frame, text=f"Value: {self.bilateral_iterations.get()}")
        self.iterations_label.pack(anchor=tk.W, padx=5)
        
        # Step 3: Final Cartoon Effect
        final_frame = ttk.LabelFrame(control_frame, text="Step 3: Final Cartoon Effect")
        final_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(final_frame, text="Edge Thickness:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Scale(final_frame, from_=1, to=5, variable=self.edge_thickness, 
                  orient=tk.HORIZONTAL, command=lambda s: self.update_label("thickness_label", f"Value: {int(float(s))}")).pack(fill=tk.X, padx=5)
        self.thickness_label = ttk.Label(final_frame, text=f"Value: {self.edge_thickness.get()}")
        self.thickness_label.pack(anchor=tk.W, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply Effect", command=self.apply_cartoon_effect).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Result", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        # Setup image display area with tabs for different processing steps
        self.notebook = ttk.Notebook(self.display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for each processing step
        self.original_tab = ttk.Frame(self.notebook)
        self.median_tab = ttk.Frame(self.notebook)
        self.edge_tab = ttk.Frame(self.notebook)
        self.color_tab = ttk.Frame(self.notebook)
        self.cartoon_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.original_tab, text="Original Image")
        self.notebook.add(self.median_tab, text="Step 1a: Median Filter")
        self.notebook.add(self.edge_tab, text="Step 1b: Edge Detection")
        self.notebook.add(self.color_tab, text="Step 2: Color Painting")
        self.notebook.add(self.cartoon_tab, text="Step 3: Final Cartoon")
        
        # Create image labels for each tab
        self.original_label = ttk.Label(self.original_tab)
        self.original_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.median_label = ttk.Label(self.median_tab)
        self.median_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.edge_label = ttk.Label(self.edge_tab)
        self.edge_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.color_label_img = ttk.Label(self.color_tab)
        self.color_label_img.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.cartoon_label = ttk.Label(self.cartoon_tab)
        self.cartoon_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    def update_label(self, label_name, text):
        if hasattr(self, label_name):
            getattr(self, label_name).config(text=text)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if not file_path:
            return
        
        try:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", "Failed to load the image.")
                return
            
            self.display_image(self.original_image, self.original_label)
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            
            # Reset other displays
            self.median_label.config(image=None)
            self.edge_label.config(image=None)
            self.color_label_img.config(image=None)
            self.cartoon_label.config(image=None)
            
            # Switch to the original image tab
            self.notebook.select(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def apply_cartoon_effect(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing is already in progress.")
            return
        
        self.is_processing = True
        self.status_var.set("Processing image...")
        
        # Start processing in a separate thread to keep the UI responsive
        self.processing_thread = threading.Thread(target=self.process_cartoon_effect)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_cartoon_effect(self):
        try:
            # Prepare parameters dictionary
            params = {
                'median_ksize': self.median_ksize.get(),
                'laplacian_ksize': self.laplacian_ksize.get(),
                'threshold': self.threshold_value.get(),
                'bilateral_d': self.bilateral_diameter.get(),
                'bilateral_color': self.bilateral_color.get(),
                'bilateral_space': self.bilateral_space.get(),
                'bilateral_iterations': self.bilateral_iterations.get(),
                'edge_thickness': self.edge_thickness.get()
            }
            
            # Process image using the backend
            result = self.backend.process_image(self.original_image, params)
            
            # Update UI with each step
            self.update_ui(result['median'], self.median_label, 1)
            self.update_ui(result['edge_mask'], self.edge_label, 2)
            self.update_ui(result['color_img'], self.color_label_img, 3)
            self.update_ui(result['cartoon'], self.cartoon_label, 4)
            
            # Update status and store result
            self.current_image = result['cartoon']
            self.status_var.set("Cartoon effect applied successfully.")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error applying cartoon effect: {str(e)}")
        
        finally:
            self.is_processing = False
    
    def update_ui(self, img, label, tab_index):
        # Convert OpenCV image to PhotoImage format for Tkinter
        if len(img.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:  # Color
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize for display if needed
        display_img = self.resize_image_for_display(img_rgb)
        
        # Convert to PhotoImage
        pil_img = Image.fromarray(display_img)
        photo_img = ImageTk.PhotoImage(image=pil_img)
        
        # Update label (must keep a reference to photo_img)
        label.config(image=photo_img)
        label.image = photo_img
        
        # Switch to the appropriate tab
        self.root.after(100, lambda: self.notebook.select(tab_index))
    
    def display_image(self, img, label):
        if img is None:
            return
        
        # Convert from BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize for display
        display_img = self.resize_image_for_display(img_rgb)
        
        # Convert to PhotoImage
        pil_img = Image.fromarray(display_img)
        photo_img = ImageTk.PhotoImage(image=pil_img)
        
        # Update label
        label.config(image=photo_img)
        label.image = photo_img  # Keep a reference
    
    def resize_image_for_display(self, img, max_width=800, max_height=600):
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Calculate aspect ratio
        aspect = w / h
        
        # Determine new dimensions while maintaining aspect ratio
        if w > max_width or h > max_height:
            if aspect > 1:  # Width > Height
                new_w = max_width
                new_h = int(new_w / aspect)
            else:  # Height > Width
                new_h = max_height
                new_w = int(new_h * aspect)
            
            # Resize the image
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return img
    
    def save_image(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "No processed image to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            cv2.imwrite(file_path, self.current_image)
            self.status_var.set(f"Image saved to: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving image: {str(e)}")
    
    def show_about(self):
        about_text = """Cartoon Effect Application

This application applies a cartoon effect to images by:
1. Detecting edges using Laplacian filtering
2. Smoothing colors using bilateral filtering
3. Combining both to create a cartoon-like effect

Adjust the parameters using the sliders to achieve 
different cartoon styles and effects.

Created for educational purposes."""

        messagebox.showinfo("About", about_text)

def main():
    root = tk.Tk()
    app = CartoonEffectApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()