import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
                            QTabWidget, QSplitter, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPalette
from ar_backend import ARBackend

class ProcessingThread(QThread):
    """Thread for processing videos without blocking the UI."""
    progress_updated = pyqtSignal(int, int)
    processing_finished = pyqtSignal(bool, str)
    
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
    
    def run(self):
        success, message = self.backend.process_videos()
        self.processing_finished.emit(success, message)

class HomographyTestThread(QThread):
    """Thread for testing homography between reference image and first video frame."""
    test_finished = pyqtSignal(bool, str, object)
    
    def __init__(self, backend, book_img_path, book_video_path):
        super().__init__()
        self.backend = backend
        self.book_img_path = book_img_path
        self.book_video_path = book_video_path
    
    def run(self):
        success, message, vis_img = self.backend.test_homography(
            self.book_img_path, self.book_video_path
        )
        self.test_finished.emit(success, message, vis_img)

class ARGUI(QMainWindow):
    """Main GUI window for AR application."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize backend
        self.backend = ARBackend()
        
        # Set up UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle("AR Video Overlay Application")
        self.setMinimumSize(1200, 800)
        
        # Set up main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Add title label
        title_label = QLabel("Augmented Reality with Planar Homographies")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #f2f2f2;
                border: 1px solid #cccccc;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #e6e6e6;
            }
        """)
        
        # Create content for tabs
        self.create_input_tab()
        self.create_preview_tab()
        self.create_processing_tab()
        
        main_layout.addWidget(self.tabs)
        
        # Add status bar
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("background-color: #f8f9fa; color: #333;")
        
        # Apply global stylesheet
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
            QLabel {
                color: #2c3e50;
            }
            QLineEdit, QComboBox {
                padding: 6px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)
        
        # Show window
        self.show()
    
    def create_input_tab(self):
        """Create the input tab for file selection."""
        input_tab = QWidget()
        layout = QVBoxLayout()
        
        # Book video selection
        book_video_layout = QHBoxLayout()
        book_video_label = QLabel("Target Video:")
        book_video_label.setMinimumWidth(150)
        self.book_video_path_label = QLabel("No file selected")
        self.book_video_path_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.book_video_path_label.setWordWrap(True)
        book_video_btn = QPushButton("Browse")
        book_video_btn.clicked.connect(self.select_book_video)
        
        book_video_layout.addWidget(book_video_label)
        book_video_layout.addWidget(self.book_video_path_label, 1)
        book_video_layout.addWidget(book_video_btn)
        
        # Book image selection
        book_img_layout = QHBoxLayout()
        book_img_label = QLabel("Reference Image:")
        book_img_label.setMinimumWidth(150)
        self.book_img_path_label = QLabel("No file selected")
        self.book_img_path_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.book_img_path_label.setWordWrap(True)
        book_img_btn = QPushButton("Browse")
        book_img_btn.clicked.connect(self.select_book_img)
        
        book_img_layout.addWidget(book_img_label)
        book_img_layout.addWidget(self.book_img_path_label, 1)
        book_img_layout.addWidget(book_img_btn)
        
        # AR source video selection
        ar_source_layout = QHBoxLayout()
        ar_source_label = QLabel("Overlay Video:")
        ar_source_label.setMinimumWidth(150)
        self.ar_source_path_label = QLabel("No file selected")
        self.ar_source_path_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.ar_source_path_label.setWordWrap(True)
        ar_source_btn = QPushButton("Browse")
        ar_source_btn.clicked.connect(self.select_ar_source)
        
        ar_source_layout.addWidget(ar_source_label)
        ar_source_layout.addWidget(self.ar_source_path_label, 1)
        ar_source_layout.addWidget(ar_source_btn)
        
        # Output video selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Video:")
        output_label.setMinimumWidth(150)
        self.output_path_label = QLabel("No file selected")
        self.output_path_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.output_path_label.setWordWrap(True)
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(self.select_output)
        
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_path_label, 1)
        output_layout.addWidget(output_btn)
        
        # Test homography button
        test_layout = QHBoxLayout()
        test_spacer = QWidget()
        test_spacer.setMinimumWidth(150)
        self.test_homography_btn = QPushButton("Test Homography")
        self.test_homography_btn.setEnabled(False)
        self.test_homography_btn.clicked.connect(self.test_homography)
        
        test_layout.addWidget(test_spacer)
        test_layout.addWidget(self.test_homography_btn)
        test_layout.addStretch(1)
        
        # Next button
        next_layout = QHBoxLayout()
        next_spacer = QWidget()
        self.next_btn = QPushButton("Next")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        
        next_layout.addStretch(1)
        next_layout.addWidget(self.next_btn)
        
        # Add layouts to main layout
        layout.addSpacing(20)
        layout.addLayout(book_video_layout)
        layout.addSpacing(10)
        layout.addLayout(book_img_layout)
        layout.addSpacing(10)
        layout.addLayout(ar_source_layout)
        layout.addSpacing(10)
        layout.addLayout(output_layout)
        layout.addSpacing(20)
        layout.addLayout(test_layout)
        layout.addSpacing(20)
        layout.addLayout(next_layout)
        layout.addStretch(1)
        
        input_tab.setLayout(layout)
        self.tabs.addTab(input_tab, "1. Input Selection")
    
    def create_preview_tab(self):
        """Create the preview tab for image visualization."""
        preview_tab = QWidget()
        layout = QVBoxLayout()
        
        # Create image displays
        display_layout = QHBoxLayout()
        
        # Book image display
        book_img_frame = QFrame()
        book_img_frame.setFrameShape(QFrame.StyledPanel)
        book_img_frame.setStyleSheet("background-color: #f8f9fa; border-radius: 5px;")
        book_img_layout = QVBoxLayout()
        book_img_title = QLabel("Reference Image")
        book_img_title.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        book_img_title.setFont(font)
        self.book_img_display = QLabel()
        self.book_img_display.setAlignment(Qt.AlignCenter)
        self.book_img_display.setMinimumSize(QSize(400, 300))
        self.book_img_display.setStyleSheet("background-color: #eaeaea; border-radius: 5px;")
        self.book_img_display.setText("Reference image will be displayed here")
        
        book_img_layout.addWidget(book_img_title)
        book_img_layout.addWidget(self.book_img_display, 1)
        book_img_frame.setLayout(book_img_layout)
        
        # AR source display
        ar_source_frame = QFrame()
        ar_source_frame.setFrameShape(QFrame.StyledPanel)
        ar_source_frame.setStyleSheet("background-color: #f8f9fa; border-radius: 5px;")
        ar_source_layout = QVBoxLayout()
        ar_source_title = QLabel("Overlay Video (First Frame)")
        ar_source_title.setAlignment(Qt.AlignCenter)
        ar_source_title.setFont(font)
        self.ar_source_display = QLabel()
        self.ar_source_display.setAlignment(Qt.AlignCenter)
        self.ar_source_display.setMinimumSize(QSize(400, 300))
        self.ar_source_display.setStyleSheet("background-color: #eaeaea; border-radius: 5px;")
        self.ar_source_display.setText("Overlay video frame will be displayed here")

        ar_source_layout.addWidget(ar_source_title)
        ar_source_layout.addWidget(self.ar_source_display, 1)
        ar_source_frame.setLayout(ar_source_layout)

        # Add displays to layout
        display_layout.addWidget(book_img_frame)
        display_layout.addWidget(ar_source_frame)

        # Homography visualization
        homography_frame = QFrame()
        homography_frame.setFrameShape(QFrame.StyledPanel)
        homography_frame.setStyleSheet("background-color: #f8f9fa; border-radius: 5px;")
        homography_layout = QVBoxLayout()
        homography_title = QLabel("Homography Visualization")
        homography_title.setAlignment(Qt.AlignCenter)
        homography_title.setFont(font)
        self.homography_display = QLabel()
        self.homography_display.setAlignment(Qt.AlignCenter)
        self.homography_display.setMinimumSize(QSize(800, 300))
        self.homography_display.setStyleSheet("background-color: #eaeaea; border-radius: 5px;")
        self.homography_display.setText("Homography matches will be displayed here")

        homography_layout.addWidget(homography_title)
        homography_layout.addWidget(self.homography_display, 1)
        homography_frame.setLayout(homography_layout)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        self.process_btn = QPushButton("Process Video")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        self.process_btn.clicked.connect(self.start_processing)

        nav_layout.addWidget(back_btn)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.process_btn)

        # Add all layouts to main layout
        layout.addLayout(display_layout)
        layout.addWidget(homography_frame)
        layout.addLayout(nav_layout)

        preview_tab.setLayout(layout)
        self.tabs.addTab(preview_tab, "2. Preview & Test")

    def create_processing_tab(self):
        """Create the processing tab for video processing progress."""
        processing_tab = QWidget()
        layout = QVBoxLayout()
        
        # Status display
        status_layout = QVBoxLayout()
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.StyledPanel)
        status_frame.setStyleSheet("background-color: #f8f9fa; border-radius: 5px; padding: 20px;")
        status_inner_layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Ready to process")
        self.status_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.status_label.setFont(font)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v/%m frames (%p%)")
        
        # Frame counter
        self.frame_counter = QLabel("0/0 frames processed")
        self.frame_counter.setAlignment(Qt.AlignCenter)
        
        # Add widgets to inner layout
        status_inner_layout.addWidget(self.status_label)
        status_inner_layout.addSpacing(20)
        status_inner_layout.addWidget(self.progress_bar)
        status_inner_layout.addWidget(self.frame_counter)
        
        status_frame.setLayout(status_inner_layout)
        status_layout.addWidget(status_frame)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        
        self.open_output_btn = QPushButton("Open Output Folder")
        self.open_output_btn.setEnabled(False)
        self.open_output_btn.clicked.connect(self.open_output_folder)
        
        self.restart_btn = QPushButton("Start Over")
        self.restart_btn.clicked.connect(self.restart)
        
        control_layout.addWidget(self.cancel_btn)
        control_layout.addStretch(1)
        control_layout.addWidget(self.open_output_btn)
        control_layout.addWidget(self.restart_btn)
        
        # Add layouts to main layout
        layout.addSpacing(40)
        layout.addLayout(status_layout)
        layout.addSpacing(40)
        layout.addLayout(control_layout)
        layout.addStretch(1)
        
        processing_tab.setLayout(layout)
        self.tabs.addTab(processing_tab, "3. Processing")

    def select_book_video(self):
        """Open file dialog to select book video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Target Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.book_video_path_label.setText(file_path)
            self.book_video_path_label.setStyleSheet("color: #2c3e50;")
            self.check_inputs()
            
            # Extract and display first frame
            frame = self.backend.extract_frame(file_path)
            if frame is not None:
                self.display_image(frame, self.book_img_display, "Target video frame")

    def select_book_img(self):
        """Open file dialog to select book reference image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "", "Image Files (*.jpg *.jpeg *.png)"
        )
        if file_path:
            self.book_img_path_label.setText(file_path)
            self.book_img_path_label.setStyleSheet("color: #2c3e50;")
            self.check_inputs()
            
            # Display image
            img = cv2.imread(file_path)
            if img is not None:
                self.display_image(img, self.book_img_display, "Reference image")

    def select_ar_source(self):
        """Open file dialog to select AR source video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Overlay Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.ar_source_path_label.setText(file_path)
            self.ar_source_path_label.setStyleSheet("color: #2c3e50;")
            self.check_inputs()
            
            # Extract and display first frame
            frame = self.backend.extract_frame(file_path)
            if frame is not None:
                self.display_image(frame, self.ar_source_display, "Overlay video frame")

    def select_output(self):
        """Open file dialog to select output file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output Video", "", "Video Files (*.mp4)"
        )
        if file_path:
            # Ensure file has .mp4 extension
            if not file_path.lower().endswith('.mp4'):
                file_path += '.mp4'
            
            self.output_path_label.setText(file_path)
            self.output_path_label.setStyleSheet("color: #2c3e50;")
            self.check_inputs()

    def check_inputs(self):
        """Check if all inputs are selected and enable buttons if they are."""
        book_video = self.book_video_path_label.text() != "No file selected"
        book_img = self.book_img_path_label.text() != "No file selected"
        ar_source = self.ar_source_path_label.text() != "No file selected"
        output = self.output_path_label.text() != "No file selected"
        
        self.test_homography_btn.setEnabled(book_video and book_img)
        self.next_btn.setEnabled(book_video and book_img and ar_source and output)

    def display_image(self, image, label, alt_text=""):
        """Display an image in a QLabel."""
        if image is None:
            label.setText(f"Failed to load {alt_text}")
            return
            
        # Convert image from BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit the label while maintaining aspect ratio
        h, w = image.shape[:2]
        label_w, label_h = label.width(), label.height()
        
        # Calculate scaling factor
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Convert to QImage and then to QPixmap
        if len(resized.shape) == 3:
            q_img = QImage(resized.data, new_w, new_h, resized.strides[0], QImage.Format_RGB888)
        else:
            q_img = QImage(resized.data, new_w, new_h, resized.strides[0], QImage.Format_Grayscale8)
            
        pixmap = QPixmap.fromImage(q_img)
        
        # Set pixmap to label
        label.setPixmap(pixmap)

    def test_homography(self):
        """Test homography between reference image and first video frame."""
        book_img_path = self.book_img_path_label.text()
        book_video_path = self.book_video_path_label.text()
        
        if book_img_path == "No file selected" or book_video_path == "No file selected":
            QMessageBox.warning(self, "Missing Input", "Please select both reference image and target video.")
            return
        
        # Disable button during test
        self.test_homography_btn.setEnabled(False)
        self.test_homography_btn.setText("Testing...")
        self.statusBar().showMessage("Testing homography...")
        
        # Start test in a separate thread
        self.homography_test_thread = HomographyTestThread(
            self.backend, book_img_path, book_video_path
        )
        self.homography_test_thread.test_finished.connect(self.on_homography_test_finished)
        self.homography_test_thread.start()

    def on_homography_test_finished(self, success, message, vis_img):
        """Handle homography test results."""
        # Re-enable button
        self.test_homography_btn.setEnabled(True)
        self.test_homography_btn.setText("Test Homography")
        
        if success:
            self.statusBar().showMessage(message)
            self.process_btn.setEnabled(self.next_btn.isEnabled())
            
            # Display visualization
            if vis_img is not None:
                self.display_image(vis_img, self.homography_display, "Homography visualization")
        else:
            self.statusBar().showMessage(f"Homography test failed: {message}")
            QMessageBox.warning(self, "Homography Test Failed", message)

    def start_processing(self):
        """Start video processing."""
        # Set paths in backend
        self.backend.set_paths(
            self.book_video_path_label.text(),
            self.book_img_path_label.text(),
            self.ar_source_path_label.text(),
            self.output_path_label.text()
        )
        
        # Set progress callback
        self.backend.set_progress_callback(self.update_progress)
        
        # Update UI
        self.status_label.setText("Processing...")
        self.progress_bar.setValue(0)
        self.frame_counter.setText("0/0 frames processed")
        self.cancel_btn.setEnabled(True)
        self.open_output_btn.setEnabled(False)
        self.statusBar().showMessage("Processing videos...")
        
        # Start processing in a separate thread
        self.processing_thread = ProcessingThread(self.backend)
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def update_progress(self, frame_num, total_frames):
        """Update progress bar and frame counter."""
        self.progress_bar.setMaximum(total_frames)
        self.progress_bar.setValue(frame_num)
        self.frame_counter.setText(f"{frame_num}/{total_frames} frames processed")

    def on_processing_finished(self, success, message):
        """Handle processing completion."""
        self.cancel_btn.setEnabled(False)
        
        if success:
            self.status_label.setText("Processing completed successfully")
            self.statusBar().showMessage("Processing completed successfully")
            self.open_output_btn.setEnabled(True)
            # Show success message
            QMessageBox.information(self, "Processing Complete",
                                  "Video processing completed successfully.\n\n"
                                  f"Output saved to: {self.output_path_label.text()}")
        else:
            self.status_label.setText("Processing failed")
            self.statusBar().showMessage(f"Processing failed: {message}")
            # Show error message
            QMessageBox.critical(self, "Processing Failed", 
                               f"Video processing failed: {message}")

    def cancel_processing(self):
        """Cancel ongoing processing."""
        self.backend.request_cancel()
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Cancelling...")
        self.statusBar().showMessage("Cancelling processing...")

    def open_output_folder(self):
        """Open the folder containing the output file."""
        output_path = self.output_path_label.text()
        if output_path != "No file selected":
            output_dir = os.path.dirname(output_path)
            # Open folder using OS-specific command
            if sys.platform == 'win32':
                os.startfile(output_dir)
            elif sys.platform == 'darwin':
                import subprocess
                subprocess.Popen(['open', output_dir])
            else:
                import subprocess
                subprocess.Popen(['xdg-open', output_dir])

    def restart(self):
        """Reset the application to initial state."""
        # Go back to first tab
        self.tabs.setCurrentIndex(0)
        
        # Reset progress UI
        self.status_label.setText("Ready to process")
        self.progress_bar.setValue(0)
        self.frame_counter.setText("0/0 frames processed")
        
        # Reset backend
        self.backend = ARBackend()
        
        # Update status
        self.statusBar().showMessage("Ready")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for a modern look
    
    # Set application icon
    # app.setWindowIcon(QIcon("icon.png"))  # Uncomment and add an icon file if available
    
    # Create and show GUI
    gui = ARGUI()
    
    sys.exit(app.exec_())