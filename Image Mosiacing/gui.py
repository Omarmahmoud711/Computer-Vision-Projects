import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, 
                            QHBoxLayout, QWidget, QFileDialog, QProgressBar, QMessageBox,
                            QScrollArea, QStatusBar, QSplitter, QListWidget, QListWidgetItem,
                            QGridLayout, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

# Import our image stitcher backend
from image_stitching_backend import ImageStitcher

class ImageStitchingThread(QThread):
    """Thread for running the image stitching process"""
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(object, bool, list)
    
    def __init__(self, images):
        super().__init__()
        self.images = images
        self.stitcher = ImageStitcher()
    
    def run(self):
        total = len(self.images) - 1
        # Start with the first image
        result = self.images[0]
        success = True
        insufficient_matches = []
        
        for i in range(1, len(self.images)):
            # Update progress
            self.progress_signal.emit(int((i / total) * 100))
            
            # Stitch the result so far with the next image
            new_result, num_matches = self.stitcher.stitch_images(self.images[i], result)
            
            if new_result is None:
                insufficient_matches.append(i)
                success = False
                continue
                
            result = new_result
            
        # Emit the result signal with the stitched image and status
        self.result_signal.emit(result, success, insufficient_matches)


class ImageStitcherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.images = []
        self.image_paths = []
        self.stitcher = ImageStitcher()
        self.result_image = None
        self.init_ui()
        
    def init_ui(self):
        # Window properties
        self.setWindowTitle('Image Mosaic Creator')
        self.setGeometry(100, 100, 1200, 800)
        
        # Set dark theme
        self.set_dark_theme()
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel - Images list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Group box for image list
        images_group = QGroupBox("Selected Images")
        images_group_layout = QVBoxLayout()
        
        # Image list
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(80, 80))
        self.image_list.setSelectionMode(QListWidget.ExtendedSelection)
        images_group_layout.addWidget(self.image_list)
        
        # Buttons for image list manipulation
        list_buttons_layout = QHBoxLayout()
        
        self.add_images_btn = QPushButton("Add Images")
        self.add_images_btn.clicked.connect(self.open_images)
        self.add_images_btn.setStyleSheet("padding: 10px;")
        
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.clicked.connect(self.remove_selected_images)
        self.remove_selected_btn.setStyleSheet("padding: 10px;")
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all_images)
        self.clear_all_btn.setStyleSheet("padding: 10px;")
        
        list_buttons_layout.addWidget(self.add_images_btn)
        list_buttons_layout.addWidget(self.remove_selected_btn)
        list_buttons_layout.addWidget(self.clear_all_btn)
        
        images_group_layout.addLayout(list_buttons_layout)
        images_group.setLayout(images_group_layout)
        
        left_layout.addWidget(images_group)
        
        # Stitching controls group
        controls_group = QGroupBox("Stitching Controls")
        controls_layout = QVBoxLayout()
        
        # Progress bar
        progress_label = QLabel("Progress:")
        self.progress_bar = QProgressBar()
        controls_layout.addWidget(progress_label)
        controls_layout.addWidget(self.progress_bar)
        
        # Stitch button
        self.stitch_btn = QPushButton("Create Mosaic")
        self.stitch_btn.clicked.connect(self.start_stitching)
        self.stitch_btn.setEnabled(False)
        self.stitch_btn.setStyleSheet("padding: 15px; font-weight: bold; font-size: 14px;")
        controls_layout.addWidget(self.stitch_btn)
        
        # Save result button
        self.save_btn = QPushButton("Save Result")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("padding: 15px;")
        controls_layout.addWidget(self.save_btn)
        
        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)
        
        # Right panel - Image display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Image view area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumWidth(600)
        
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setText("Load images and create a mosaic to see the result here.")
        self.image_display.setStyleSheet("background-color: #2a2a2a;")
        
        self.scroll_area.setWidget(self.image_display)
        right_layout.addWidget(self.scroll_area)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to load images")
        
        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def set_dark_theme(self):
        """Apply dark theme to the application"""
        palette = QPalette()
        
        # Dark palette colors
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        
        QApplication.setPalette(palette)
        
        # Set style for QGroupBox
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                background-color: #444;
                border: none;
                border-radius: 4px;
                color: white;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #666;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
            QProgressBar {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2a82da;
                width: 10px;
            }
        """)
    
    def open_images(self):
        """Open file dialog to select images"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Image Files", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not file_paths:
            return
            
        # Clear previous images if any
        self.status_bar.showMessage(f"Loading {len(file_paths)} images...")
        
        # Load and add images to the list
        for file_path in file_paths:
            img = cv2.imread(file_path)
            if img is None:
                self.status_bar.showMessage(f"Failed to load {os.path.basename(file_path)}")
                continue
                
            # Add image to our list
            self.images.append(img)
            self.image_paths.append(file_path)
            
            # Create thumbnail for list item
            thumbnail = cv2.resize(img, (80, 80))
            h, w, c = thumbnail.shape
            q_img = QImage(thumbnail.data, w, h, w * c, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Add to list widget
            item = QListWidgetItem(os.path.basename(file_path))
            item.setIcon(QIcon(pixmap))
            self.image_list.addItem(item)
        
        # Enable stitch button if we have at least 2 images
        self.stitch_btn.setEnabled(len(self.images) >= 2)
        self.status_bar.showMessage(f"Loaded {len(file_paths)} images. Total: {len(self.images)}")
    
    def remove_selected_images(self):
        """Remove selected images from the list"""
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            return
            
        for item in selected_items:
            index = self.image_list.row(item)
            self.image_list.takeItem(index)
            self.images.pop(index)
            self.image_paths.pop(index)
        
        # Update stitch button state
        self.stitch_btn.setEnabled(len(self.images) >= 2)
        self.status_bar.showMessage(f"Removed {len(selected_items)} images. Total: {len(self.images)}")
    
    def clear_all_images(self):
        """Clear all images from the list"""
        self.image_list.clear()
        self.images.clear()
        self.image_paths.clear()
        self.stitch_btn.setEnabled(False)
        self.status_bar.showMessage("All images cleared")
    
    def start_stitching(self):
        """Start the stitching process in a separate thread"""
        if len(self.images) < 2:
            QMessageBox.warning(self, "Warning", "Need at least 2 images to create a mosaic.")
            return
            
        # Disable controls during processing
        self.stitch_btn.setEnabled(False)
        self.add_images_btn.setEnabled(False)
        self.remove_selected_btn.setEnabled(False)
        self.clear_all_btn.setEnabled(False)
        
        self.status_bar.showMessage("Stitching images...")
        
        # Initialize and start the worker thread
        self.stitch_thread = ImageStitchingThread(self.images)
        self.stitch_thread.progress_signal.connect(self.update_progress)
        self.stitch_thread.result_signal.connect(self.process_result)
        self.stitch_thread.start()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def process_result(self, result_image, success, insufficient_matches):
        """Process the stitching result"""
        # Re-enable controls
        self.stitch_btn.setEnabled(True)
        self.add_images_btn.setEnabled(True)
        self.remove_selected_btn.setEnabled(True)
        self.clear_all_btn.setEnabled(True)
        
        # Check if stitching was successful
        if not success:
            # Format the list of images with insufficient matches
            bad_images = [f"Image {i+1}" for i in insufficient_matches]
            QMessageBox.warning(self, "Stitching Warning", 
                               f"The following images didn't have enough correspondences:\n" +
                               "\n".join(bad_images))
        
        # If we have a result image
        if result_image is not None:
            self.result_image = result_image
            self.display_result()
            self.save_btn.setEnabled(True)
            self.status_bar.showMessage("Mosaic created successfully!")
        else:
            self.status_bar.showMessage("Failed to create mosaic")
    
    def display_result(self):
        """Display the stitched image result"""
        if self.result_image is None:
            return
            
        # Convert the result image to QImage
        h, w, c = self.result_image.shape
        bytes_per_line = c * w
        q_img = QImage(self.result_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        
        # Scale down if the image is too large for display
        screen_rect = QApplication.desktop().screenGeometry()
        max_width = screen_rect.width() * 0.8
        max_height = screen_rect.height() * 0.8
        
        scaled_pixmap = QPixmap.fromImage(q_img)
        if w > max_width or h > max_height:
            scaled_pixmap = scaled_pixmap.scaled(
                int(max_width), int(max_height), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        
        # Display the image
        self.image_display.setPixmap(scaled_pixmap)
    
    def save_result(self):
        """Save the stitched image to disk"""
        if self.result_image is None:
            return
            
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mosaic", "", "PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp)"
        )
        
        if not file_path:
            return
            
        # Save the image
        success = cv2.imwrite(file_path, self.result_image)
        if success:
            self.status_bar.showMessage(f"Mosaic saved to {file_path}")
        else:
            self.status_bar.showMessage("Failed to save mosaic")
            QMessageBox.critical(self, "Error", "Failed to save the mosaic image.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageStitcherApp()
    window.show()
    sys.exit(app.exec_())