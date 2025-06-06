import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QLabel, QSlider, QHBoxLayout, QFileDialog, QMessageBox, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent, QMouseEvent, QPalette, QColor, QPainter
import cv2
import numpy as np

class OutputWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projection Output")
        self.setGeometry(100, 100, 800, 600)
        
        # Track window states
        self.is_fullscreen = False
        self.is_borderless = True
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        # For window dragging
        self.dragging = False
        self.drag_position = QPoint()
        
        # Multiple videos support
        self.videos = []  # List of video paths
        self.video_captures = {}  # Dictionary of video captures
        self.video_timers = {}  # Dictionary of video timers
        self.video_frames = {}  # Dictionary to store current frames
        self.video_settings = {}  # Dictionary to store individual video settings
        self.is_playing = False
        
        # Scale factor
        self.scale_factor = 1.0
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        
        # Set black background
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        self.setPalette(palette)
        central_widget.setAutoFillBackground(True)
        
        # Create label for displaying the output
        self.output_label = QLabel()
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.output_label)
        
        # Initialize with a black image
        self.original_image = np.zeros((600, 800, 3), dtype=np.uint8)
        self.update_output(self.original_image)

    def apply_warp(self, image, pitch, yaw, keystone_h, keystone_v, pos_x, pos_y):
        if image is None or image.size == 0:
            return image

        height, width = image.shape[:2]
        
        # Convert angles to radians
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        # Create transformation matrix
        # For pitch (around X-axis)
        pitch_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        
        # For yaw (around Y-axis)
        yaw_matrix = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])
        
        # Combine transformations
        transform = np.dot(yaw_matrix, pitch_matrix)
        
        # Create source points (corners of the image)
        src_points = np.float32([
            [0, 0],
            [width-1, 0],
            [0, height-1],
            [width-1, height-1]
        ])
        
        # Calculate destination points after transformation
        dst_points = []
        for x, y in src_points:
            # Convert to homogeneous coordinates
            point = np.array([x - width/2, y - height/2, 0])
            # Apply transformation
            transformed = np.dot(transform, point)
            # Convert back from homogeneous coordinates
            dst_points.append([transformed[0] + width/2, transformed[1] + height/2])
        
        dst_points = np.float32(dst_points)
        
        # Apply horizontal keystone adjustment
        if keystone_h != 0:
            keystone_factor = keystone_h / 100.0
            shift = width * keystone_factor * 0.5
            dst_points[0][0] -= shift
            dst_points[1][0] += shift
        
        # Apply vertical keystone adjustment
        if keystone_v != 0:
            keystone_factor = keystone_v / 100.0
            shift = height * keystone_factor * 0.5
            dst_points[0][1] -= shift
            dst_points[2][1] += shift
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply the transformation
        warped = cv2.warpPerspective(image, matrix, (width, height), 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(0, 0, 0))
        
        # Apply position offset
        if pos_x != 0 or pos_y != 0:
            translation_matrix = np.float32([[1, 0, pos_x], [0, 1, pos_y]])
            warped = cv2.warpAffine(warped, translation_matrix, (width, height),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
        
        return warped

    def update_output(self, image):
        self.original_image = image.copy()
        self.update_warped_output()

    def update_warped_output(self):
        if hasattr(self, 'original_image'):
            # Get current warp values from control window
            if hasattr(self, 'control_window'):
                pitch = self.control_window.pitch_slider.value()
                yaw = self.control_window.yaw_slider.value()
                keystone_h = self.control_window.keystone_h_slider.value()
                keystone_v = self.control_window.keystone_v_slider.value()
                pos_x = self.control_window.pos_x_slider.value()
                pos_y = self.control_window.pos_y_slider.value()
                self.scale_factor = self.control_window.scale_slider.value() / 100.0
                warped_image = self.apply_warp(self.original_image, pitch, yaw, 
                                            keystone_h, keystone_v, pos_x, pos_y)
            else:
                warped_image = self.original_image

            # Convert numpy array to QImage
            height, width, channel = warped_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(warped_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Convert QImage to QPixmap and display
            pixmap = QPixmap.fromImage(q_image)
            
            # Calculate the scaled size while maintaining aspect ratio
            label_size = self.output_label.size()
            
            # Calculate the scaled dimensions based on the original image size and scale factor
            original_aspect = width / height
            if original_aspect > 1:  # Wider than tall
                scaled_width = int(width * self.scale_factor)
                scaled_height = int(scaled_width / original_aspect)
            else:  # Taller than wide
                scaled_height = int(height * self.scale_factor)
                scaled_width = int(scaled_height * original_aspect)
            
            scaled_pixmap = pixmap.scaled(
                scaled_width, scaled_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Create a black background pixmap
            background = QPixmap(label_size)
            background.fill(Qt.GlobalColor.black)
            
            # Calculate position to center the scaled pixmap
            x = (label_size.width() - scaled_pixmap.width()) // 2
            y = (label_size.height() - scaled_pixmap.height()) // 2
            
            # Draw the scaled pixmap on the black background
            painter = QPainter(background)
            painter.drawPixmap(x, y, scaled_pixmap)
            painter.end()
            
            self.output_label.setPixmap(background)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_F12:
            self.toggle_borderless()
        elif event.key() == Qt.Key.Key_Escape and self.is_fullscreen:
            self.toggle_fullscreen()
        super().keyPressEvent(event)

    def toggle_borderless(self):
        if self.is_fullscreen:
            return  # Don't toggle if in fullscreen
        
        self.is_borderless = not self.is_borderless
        if self.is_borderless:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        else:
            self.setWindowFlags(Qt.WindowType.Window)
        self.show()

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.showNormal()
            if self.is_borderless:
                self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
            self.show()
        else:
            self.showFullScreen()
        self.is_fullscreen = not self.is_fullscreen

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.is_borderless and not self.is_fullscreen:
            self.dragging = True
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging and self.is_borderless and not self.is_fullscreen:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            event.accept()

    def add_video(self, file_path):
        if file_path not in self.videos:
            print(f"Adding new video: {file_path}")
            self.videos.append(file_path)
            video_id = len(self.videos) - 1
            print(f"Assigned video_id: {video_id}")
            
            # Create video capture
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                print(f"Video capture opened successfully for video {video_id}")
                self.video_captures[video_id] = cap
                self.video_frames[video_id] = None
                
                # Initialize video settings
                self.video_settings[video_id] = {
                    'scale': 1.0,
                    'pos_x': 0,
                    'pos_y': 0
                }
                print(f"Initialized settings for video {video_id}")
                
                # Create timer for this video
                timer = QTimer()
                timer.timeout.connect(lambda vid=video_id: self.update_video_frame(vid))
                self.video_timers[video_id] = timer
                print(f"Created timer for video {video_id}")
                
                # Start playback if we're in playing state
                if self.is_playing:
                    timer.start(33)  # ~30 FPS
                    print(f"Started timer for video {video_id}")
                return True
            else:
                print(f"Failed to open video capture for {file_path}")
        else:
            print(f"Video already exists: {file_path}")
        return False

    def remove_video(self, index):
        print(f"Removing video at index {index}")
        if 0 <= index < len(self.videos):
            # Stop and cleanup the video
            if index in self.video_timers:
                print(f"Stopping timer for video {index}")
                self.video_timers[index].stop()
                del self.video_timers[index]
            if index in self.video_captures:
                print(f"Releasing capture for video {index}")
                self.video_captures[index].release()
                del self.video_captures[index]
            if index in self.video_frames:
                print(f"Removing frame for video {index}")
                del self.video_frames[index]
            
            # Remove from videos list
            print(f"Removing video path: {self.videos[index]}")
            self.videos.pop(index)
            
            # Update remaining video IDs
            print("Updating remaining video IDs")
            new_captures = {}
            new_timers = {}
            new_frames = {}
            for i, video_id in enumerate(self.videos):
                if video_id in self.video_captures:
                    new_captures[i] = self.video_captures[video_id]
                    new_timers[i] = self.video_timers[i]
                    new_frames[i] = self.video_frames[i]
            
            self.video_captures = new_captures
            self.video_timers = new_timers
            self.video_frames = new_frames
            
            # Update display
            print("Updating combined output")
            self.update_combined_output()
            return True
        print(f"Invalid video index: {index}")
        return False

    def update_video_frame(self, video_id):
        if video_id in self.video_captures and self.is_playing:
            cap = self.video_captures[video_id]
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.video_frames[video_id] = frame
                self.update_combined_output()
            else:
                # End of video, restart
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.video_frames[video_id] = frame
                    self.update_combined_output()

    def update_video_settings(self, video_id, scale=None, pos_x=None, pos_y=None):
        if video_id in self.video_settings:
            if scale is not None:
                self.video_settings[video_id]['scale'] = scale
            if pos_x is not None:
                self.video_settings[video_id]['pos_x'] = pos_x
            if pos_y is not None:
                self.video_settings[video_id]['pos_y'] = pos_y
            self.update_combined_output()

    def update_combined_output(self):
        if not self.video_frames:
            # No videos, show black screen
            self.update_output(np.zeros((600, 800, 3), dtype=np.uint8))
            return

        # Create a black canvas
        combined_frame = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Place each video according to its settings
        for video_id, frame in self.video_frames.items():
            if frame is not None and video_id in self.video_settings:
                settings = self.video_settings[video_id]
                
                # Get original dimensions
                height, width = frame.shape[:2]
                
                # Calculate scaled dimensions
                scaled_width = int(width * settings['scale'])
                scaled_height = int(height * settings['scale'])
                
                # Resize frame with better quality
                resized = cv2.resize(frame, (scaled_width, scaled_height), 
                                   interpolation=cv2.INTER_LANCZOS4)
                
                # Calculate position
                x = settings['pos_x']
                y = settings['pos_y']
                
                # Ensure the video stays within bounds
                x = max(0, min(x, 800 - scaled_width))
                y = max(0, min(y, 600 - scaled_height))
                
                # Create a mask for the resized frame
                mask = np.ones_like(resized)
                
                # Place the frame on the combined frame with alpha blending
                try:
                    # Calculate the region of interest
                    roi = combined_frame[y:y+scaled_height, x:x+scaled_width]
                    
                    # If the video is partially outside the frame, adjust the ROI
                    if x < 0 or y < 0 or x + scaled_width > 800 or y + scaled_height > 600:
                        # Calculate the visible portion of the video
                        visible_x = max(0, -x)
                        visible_y = max(0, -y)
                        visible_width = min(scaled_width, 800 - x)
                        visible_height = min(scaled_height, 600 - y)
                        
                        # Adjust the video frame and ROI
                        resized = resized[visible_y:visible_y+visible_height, 
                                        visible_x:visible_x+visible_width]
                        roi = combined_frame[y+visible_y:y+visible_y+visible_height,
                                           x+visible_x:x+visible_x+visible_width]
                    
                    # Blend the video with the background
                    alpha = 0.7  # Adjust this value to control blending
                    combined_frame[y:y+scaled_height, x:x+scaled_width] = \
                        cv2.addWeighted(roi, 1-alpha, resized, alpha, 0)
                        
                except ValueError as e:
                    print(f"Error placing video {video_id}: {e}")
                    continue
        
        self.update_output(combined_frame)

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        for timer in self.video_timers.values():
            if self.is_playing:
                timer.start(33)  # ~30 FPS
            else:
                timer.stop()

    def stop_video(self):
        self.is_playing = False
        for timer in self.video_timers.values():
            timer.stop()
        for cap in self.video_captures.values():
            cap.release()
        self.video_captures.clear()
        self.video_timers.clear()
        self.video_frames.clear()
        self.videos.clear()
        # Reset to black screen
        self.update_output(np.zeros((600, 800, 3), dtype=np.uint8))

class ControlWindow(QMainWindow):
    def __init__(self, output_window):
        super().__init__()
        self.output_window = output_window
        self.output_window.control_window = self
        self.setWindowTitle("Control Panel")
        self.setGeometry(950, 100, 400, 800)  # Made window taller to accommodate more videos
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Set black background
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        self.setPalette(palette)
        central_widget.setAutoFillBackground(True)
        
        # Style for labels and buttons
        style = """
            QLabel { color: white; }
            QPushButton { 
                background-color: #333333;
                color: white;
                border: 1px solid #555555;
                padding: 5px;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #333333;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #666666;
                border: 1px solid #555555;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """
        self.setStyleSheet(style)
        
        # Add warp controls
        warp_group = QWidget()
        warp_layout = QVBoxLayout(warp_group)
        
        # Pitch control
        pitch_layout = QHBoxLayout()
        pitch_label = QLabel("Pitch")
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setMinimum(-45)
        self.pitch_slider.setMaximum(45)
        self.pitch_slider.setValue(0)
        self.pitch_slider.valueChanged.connect(self.update_warp)
        pitch_reset = QPushButton("Reset")
        pitch_reset.clicked.connect(lambda: self.reset_slider(self.pitch_slider))
        pitch_layout.addWidget(pitch_label)
        pitch_layout.addWidget(self.pitch_slider)
        pitch_layout.addWidget(pitch_reset)
        warp_layout.addLayout(pitch_layout)
        
        # Yaw control
        yaw_layout = QHBoxLayout()
        yaw_label = QLabel("Yaw")
        self.yaw_slider = QSlider(Qt.Orientation.Horizontal)
        self.yaw_slider.setMinimum(-45)
        self.yaw_slider.setMaximum(45)
        self.yaw_slider.setValue(0)
        self.yaw_slider.valueChanged.connect(self.update_warp)
        yaw_reset = QPushButton("Reset")
        yaw_reset.clicked.connect(lambda: self.reset_slider(self.yaw_slider))
        yaw_layout.addWidget(yaw_label)
        yaw_layout.addWidget(self.yaw_slider)
        yaw_layout.addWidget(yaw_reset)
        warp_layout.addLayout(yaw_layout)
        
        # Horizontal Keystone control
        keystone_h_layout = QHBoxLayout()
        keystone_h_label = QLabel("Horizontal Keystone")
        self.keystone_h_slider = QSlider(Qt.Orientation.Horizontal)
        self.keystone_h_slider.setMinimum(-100)
        self.keystone_h_slider.setMaximum(100)
        self.keystone_h_slider.setValue(0)
        self.keystone_h_slider.valueChanged.connect(self.update_warp)
        keystone_h_reset = QPushButton("Reset")
        keystone_h_reset.clicked.connect(lambda: self.reset_slider(self.keystone_h_slider))
        keystone_h_layout.addWidget(keystone_h_label)
        keystone_h_layout.addWidget(self.keystone_h_slider)
        keystone_h_layout.addWidget(keystone_h_reset)
        warp_layout.addLayout(keystone_h_layout)
        
        # Vertical Keystone control
        keystone_v_layout = QHBoxLayout()
        keystone_v_label = QLabel("Vertical Keystone")
        self.keystone_v_slider = QSlider(Qt.Orientation.Horizontal)
        self.keystone_v_slider.setMinimum(-100)
        self.keystone_v_slider.setMaximum(100)
        self.keystone_v_slider.setValue(0)
        self.keystone_v_slider.valueChanged.connect(self.update_warp)
        keystone_v_reset = QPushButton("Reset")
        keystone_v_reset.clicked.connect(lambda: self.reset_slider(self.keystone_v_slider))
        keystone_v_layout.addWidget(keystone_v_label)
        keystone_v_layout.addWidget(self.keystone_v_slider)
        keystone_v_layout.addWidget(keystone_v_reset)
        warp_layout.addLayout(keystone_v_layout)
        
        layout.addWidget(warp_group)
        
        # Add position controls
        position_group = QWidget()
        position_layout = QVBoxLayout(position_group)
        
        # X Position control
        pos_x_layout = QHBoxLayout()
        pos_x_label = QLabel("X Position")
        self.pos_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.pos_x_slider.setMinimum(-500)
        self.pos_x_slider.setMaximum(500)
        self.pos_x_slider.setValue(0)
        self.pos_x_slider.valueChanged.connect(self.update_warp)
        pos_x_reset = QPushButton("Reset")
        pos_x_reset.clicked.connect(lambda: self.reset_slider(self.pos_x_slider))
        pos_x_layout.addWidget(pos_x_label)
        pos_x_layout.addWidget(self.pos_x_slider)
        pos_x_layout.addWidget(pos_x_reset)
        position_layout.addLayout(pos_x_layout)
        
        # Y Position control
        pos_y_layout = QHBoxLayout()
        pos_y_label = QLabel("Y Position")
        self.pos_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.pos_y_slider.setMinimum(-500)
        self.pos_y_slider.setMaximum(500)
        self.pos_y_slider.setValue(0)
        self.pos_y_slider.valueChanged.connect(self.update_warp)
        pos_y_reset = QPushButton("Reset")
        pos_y_reset.clicked.connect(lambda: self.reset_slider(self.pos_y_slider))
        pos_y_layout.addWidget(pos_y_label)
        pos_y_layout.addWidget(self.pos_y_slider)
        pos_y_layout.addWidget(pos_y_reset)
        position_layout.addLayout(pos_y_layout)
        
        layout.addWidget(position_group)
        
        # Add video controls directly in the main layout
        self.video_group = QWidget()
        self.video_layout = QVBoxLayout(self.video_group)
        self.video_layout.setSpacing(10)
        self.video_layout.setContentsMargins(5, 5, 5, 5)
        
        # Video list
        self.video_list_label = QLabel("Loaded Videos:")
        self.video_layout.addWidget(self.video_list_label)
        
        # Video buttons
        video_buttons_layout = QHBoxLayout()
        
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video)
        video_buttons_layout.addWidget(self.load_video_btn)
        
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        video_buttons_layout.addWidget(self.play_pause_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_video)
        video_buttons_layout.addWidget(self.stop_btn)
        
        self.video_layout.addLayout(video_buttons_layout)
        
        # Add video list display with controls
        self.video_controls = {}  # Dictionary to store video control widgets
        self.update_video_list_display()
        
        # Add the video group directly to the main layout
        layout.addWidget(self.video_group)
        
        # Add a separator
        separator = QWidget()
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: #555555;")
        layout.addWidget(separator)
        
        # Add brightness control
        brightness_layout = QHBoxLayout()
        brightness_label = QLabel("Brightness")
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setMinimum(0)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(50)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        brightness_reset = QPushButton("Reset")
        brightness_reset.clicked.connect(lambda: self.reset_slider(self.brightness_slider, 50))
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(self.brightness_slider)
        brightness_layout.addWidget(brightness_reset)
        layout.addLayout(brightness_layout)
        
        # Add a test pattern button
        self.test_pattern_btn = QPushButton("Show Test Pattern")
        self.test_pattern_btn.clicked.connect(self.show_test_pattern)
        layout.addWidget(self.test_pattern_btn)
        
        # Add scale control
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Scale")
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(10)  # 10% of original size
        self.scale_slider.setMaximum(100)  # 100% of original size
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(self.update_warp)
        scale_reset = QPushButton("Reset")
        scale_reset.clicked.connect(lambda: self.reset_slider(self.scale_slider, 100))
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_slider)
        scale_layout.addWidget(scale_reset)
        layout.addLayout(scale_layout)
        
        # Add a test pattern button
        self.test_pattern_btn = QPushButton("Show Test Pattern")
        self.test_pattern_btn.clicked.connect(self.show_test_pattern)
        layout.addWidget(self.test_pattern_btn)
        
        # Add some spacing
        layout.addStretch()

    def update_warp(self):
        self.output_window.update_warped_output()

    def update_brightness(self, value):
        # Create a test image with the current brightness
        image = np.ones((600, 800, 3), dtype=np.uint8) * value
        self.output_window.update_output(image)

    def show_test_pattern(self):
        # Create a simple test pattern
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        # Draw some shapes
        cv2.rectangle(image, (100, 100), (700, 500), (0, 255, 0), 2)
        cv2.circle(image, (400, 300), 100, (0, 0, 255), -1)
        cv2.putText(image, "Test Pattern", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.output_window.update_output(image)

    def reset_slider(self, slider, default_value=0):
        slider.setValue(default_value)
        self.update_warp()

    def create_video_controls(self, video_id, video_path):
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        # Video name with better formatting
        name_label = QLabel(f"Video {video_id+1}: {video_path.split('/')[-1]}")
        name_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        video_layout.addWidget(name_label)
        
        # Scale control with percentage display
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Scale:")
        scale_slider = QSlider(Qt.Orientation.Horizontal)
        scale_slider.setMinimum(10)
        scale_slider.setMaximum(200)
        scale_slider.setValue(int(self.output_window.video_settings[video_id]['scale'] * 100))
        scale_value_label = QLabel(f"{scale_slider.value()}%")
        scale_slider.valueChanged.connect(
            lambda v, vid=video_id, label=scale_value_label: self.update_scale_label(v, label, vid)
        )
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(scale_slider)
        scale_layout.addWidget(scale_value_label)
        video_layout.addLayout(scale_layout)
        
        # Position controls with value display
        pos_layout = QHBoxLayout()
        
        # X position
        x_label = QLabel("X:")
        x_slider = QSlider(Qt.Orientation.Horizontal)
        x_slider.setMinimum(-400)
        x_slider.setMaximum(400)
        x_slider.setValue(self.output_window.video_settings[video_id]['pos_x'])
        x_value_label = QLabel(str(x_slider.value()))
        x_slider.valueChanged.connect(
            lambda v, vid=video_id, label=x_value_label: self.update_position_label(v, label, vid, 'x')
        )
        pos_layout.addWidget(x_label)
        pos_layout.addWidget(x_slider)
        pos_layout.addWidget(x_value_label)
        
        # Y position
        y_label = QLabel("Y:")
        y_slider = QSlider(Qt.Orientation.Horizontal)
        y_slider.setMinimum(-300)
        y_slider.setMaximum(300)
        y_slider.setValue(self.output_window.video_settings[video_id]['pos_y'])
        y_value_label = QLabel(str(y_slider.value()))
        y_slider.valueChanged.connect(
            lambda v, vid=video_id, label=y_value_label: self.update_position_label(v, label, vid, 'y')
        )
        pos_layout.addWidget(y_label)
        pos_layout.addWidget(y_slider)
        pos_layout.addWidget(y_value_label)
        
        video_layout.addLayout(pos_layout)
        
        # Add remove button with confirmation
        remove_btn = QPushButton("Remove")
        remove_btn.setStyleSheet("background-color: #ff4444;")
        remove_btn.clicked.connect(lambda checked, idx=video_id: self.confirm_remove_video(idx))
        video_layout.addWidget(remove_btn)
        
        # Add separator
        separator = QWidget()
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: #555555;")
        video_layout.addWidget(separator)
        
        return video_widget

    def update_scale_label(self, value, label, video_id):
        label.setText(f"{value}%")
        self.output_window.update_video_settings(video_id, scale=value/100.0)

    def update_position_label(self, value, label, video_id, axis):
        label.setText(str(value))
        if axis == 'x':
            self.output_window.update_video_settings(video_id, pos_x=value)
        else:
            self.output_window.update_video_settings(video_id, pos_y=value)

    def confirm_remove_video(self, index):
        reply = QMessageBox.question(
            self,
            'Confirm Removal',
            f'Are you sure you want to remove Video {index+1}?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.remove_video(index)

    def update_video_list_display(self):
        print(f"Updating video list display. Current videos: {self.output_window.videos}")
        
        # Get current video IDs
        current_video_ids = set(self.video_controls.keys())
        new_video_ids = set(range(len(self.output_window.videos)))
        
        # Remove controls for videos that no longer exist
        for video_id in current_video_ids - new_video_ids:
            print(f"Removing controls for video {video_id}")
            if video_id in self.video_controls:
                widget = self.video_controls[video_id]
                if widget.parent():
                    widget.parent().deleteLater()
                del self.video_controls[video_id]
        
        # Update or add controls for existing videos
        for i, video_path in enumerate(self.output_window.videos):
            if i in self.video_controls:
                # Update existing control
                print(f"Updating controls for video {i}")
                widget = self.video_controls[i]
                # Update the video name label
                name_label = widget.findChild(QLabel)
                if name_label:
                    name_label.setText(f"Video {i+1}: {video_path.split('/')[-1]}")
            else:
                # Create new control
                print(f"Creating new controls for video {i}")
                video_widget = self.create_video_controls(i, video_path)
                self.video_controls[i] = video_widget
                self.video_layout.addWidget(video_widget)
                print(f"Added video widget {i} to layout")
        
        if not self.output_window.videos:
            self.video_list_label.setText("No videos loaded")
            print("No videos loaded")
        else:
            self.video_list_label.setText(f"Loaded Videos: {len(self.output_window.videos)}")
            print(f"Updated display for {len(self.output_window.videos)} videos")

    def remove_video(self, index):
        if self.output_window.remove_video(index):
            self.update_video_list_display()
            # Update remaining video IDs
            for i, video_path in enumerate(self.output_window.videos):
                if i in self.video_controls:
                    self.video_controls[i].findChild(QLabel).setText(f"Video {i+1}: {video_path.split('/')[-1]}")

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if file_path:
            print(f"Loading video: {file_path}")
            if self.output_window.add_video(file_path):
                print("Video added successfully")
                self.play_pause_btn.setEnabled(True)
                self.stop_btn.setEnabled(True)
                self.update_video_list_display()
                # Start playback automatically when first video is loaded
                if len(self.output_window.videos) == 1:
                    print("Starting playback for first video")
                    self.output_window.is_playing = True
                    self.play_pause_btn.setText("Pause")
                    for timer in self.output_window.video_timers.values():
                        timer.start(33)
            else:
                print("Failed to add video")

    def toggle_playback(self):
        self.output_window.toggle_playback()
        # Update button text
        if self.output_window.is_playing:
            self.play_pause_btn.setText("Pause")
        else:
            self.play_pause_btn.setText("Play")

    def stop_video(self):
        self.output_window.stop_video()
        self.play_pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.update_video_list_display()

def main():
    app = QApplication(sys.argv)
    
    # Create and show the output window
    output_window = OutputWindow()
    output_window.show()
    
    # Create and show the control window
    control_window = ControlWindow(output_window)
    control_window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 