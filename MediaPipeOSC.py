import cv2
import mediapipe as mp
import numpy as np
from pythonosc import udp_client
import time

class ArmTracker:
    """
    A class that tracks arm movements using your camera and sends the data 
    to other programs via OSC (Open Sound Control) messages.
    
    What this does:
    - Uses your webcam to detect your body pose
    - Tracks specific points on your arms (shoulders, elbows, wrists)
    - Sends the position data to other software that can receive OSC messages
    """
    
    def __init__(self, osc_ip="127.0.0.1", osc_port=5005):
        """
        Initialize the arm tracker.
        
        Args:
            osc_ip (str): IP address to send OSC messages to
                         "127.0.0.1" means your own computer (localhost)
            osc_port (int): Port number for OSC communication
        """
        print("🚀 Starting MediaPipe Arm Tracker...")
        
        # ============ MEDIAPIPE SETUP ============
        # MediaPipe is Google's library for detecting human poses
        self.mp_pose = mp.solutions.pose
        
        # Configure the pose detection settings
        # 🔧 CHANGE THESE if you want different detection behavior:
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,          # False = video mode (not single images)
            model_complexity=1,               # 0=fast/less accurate, 1=balanced, 2=slow/more accurate
            smooth_landmarks=True,            # Reduces jittery movements
            min_detection_confidence=0.7,     # How confident to be before detecting a person (0.0-1.0)
            min_tracking_confidence=0.5       # How confident to be when tracking (0.0-1.0)
        )
        
        # For drawing the skeleton on the video
        self.mp_draw = mp.solutions.drawing_utils
        
        # ============ OSC SETUP ============
        # OSC lets us send data to other programs (like music software, games, etc.)
        # We'll send left arm data to port 8001 and right arm data to port 8002
        self.left_osc_client = udp_client.SimpleUDPClient(osc_ip, 8001)
        self.right_osc_client = udp_client.SimpleUDPClient(osc_ip, 8002)
        print(f"📡 OSC clients ready:")
        print(f"   Left arm data  -> {osc_ip}:8001")
        print(f"   Right arm data -> {osc_ip}:8002")
        
        # ============ BODY POINTS WE'RE TRACKING ============
        # Tracking left and right arm chains (shoulder -> elbow -> wrist)
        # 🔧 ADD OR REMOVE body parts here if you want to track different things:
        self.landmarks = {
            'left_shoulder': 11,   # Left shoulder joint
            'right_shoulder': 12,  # Right shoulder joint
            'left_elbow': 13,      # Left elbow joint
            'right_elbow': 14,     # Right elbow joint
            'left_wrist': 15,      # Left wrist
            'right_wrist': 16      # Right wrist
        }
        
        # ============ SMOOTHING SETUP ============
        # Raw tracking data can be jittery, so we smooth it out
        self.prev_landmarks = {}              # Stores previous positions
        self.smoothing_factor = 0.7           # 🔧 CHANGE THIS: 0.0=no smoothing, 0.9=very smooth
        
        # ============ CAMERA SETUP ============
        # 🔧 CHANGE THESE camera settings if needed:
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fps = 30
        
    def smooth_coordinates(self, landmark_name, current_coords):
        """
        Smooths out jittery movements by blending current position with previous position.
        Now works with X,Y coordinates only.
        
        🔧 TO CHANGE SMOOTHING BEHAVIOR:
        - Increase self.smoothing_factor (in __init__) for smoother but slower response
        - Decrease it for faster but more jittery response
        
        Args:
            landmark_name (str): Name of the body part (e.g., 'left_wrist')
            current_coords (list): Current [x, y] coordinates
            
        Returns:
            list: Smoothed [x, y] coordinates
        """
        # If this is the first time we see this landmark, just use current position
        if landmark_name not in self.prev_landmarks:
            self.prev_landmarks[landmark_name] = current_coords
            return current_coords
        
        # Blend previous position with current position
        # Formula: new_position = (old_position * smoothing) + (current_position * (1-smoothing))
        prev = self.prev_landmarks[landmark_name]
        smoothed = []
        
        for i in range(2):  # For x, y coordinates only
            smoothed_coord = (prev[i] * self.smoothing_factor + 
                            current_coords[i] * (1 - self.smoothing_factor))
            smoothed.append(smoothed_coord)
        
        # Store this smoothed position for next time
        self.prev_landmarks[landmark_name] = smoothed
        return smoothed
    
    def send_arm_data(self, side, shoulder_coords, elbow_coords, wrist_coords):
        """
        Sends complete arm data (shoulder, elbow, wrist) as a single OSC message.
        
        🔧 TO CHANGE OSC MESSAGE FORMAT:
        - Modify the address format
        - Change the data structure being sent
        
        Args:
            side (str): 'left' or 'right'
            shoulder_coords (list): [x, y] normalized coordinates
            elbow_coords (list): [x, y] normalized coordinates  
            wrist_coords (list): [x, y] normalized coordinates
        """
        try:
            # Combine all 6 values into one message: [shoulderX, shoulderY, elbowX, elbowY, wristX, wristY]
            arm_data = shoulder_coords + elbow_coords + wrist_coords
            
            # Send to appropriate port based on which arm
            if side == 'left':
                self.left_osc_client.send_message("/arm/left", arm_data)
            else:
                self.right_osc_client.send_message("/arm/right", arm_data)
                
        except Exception as e:
            print(f"❌ Error sending {side} arm OSC message: {e}")
    
    def send_osc_message(self, address, values):
        """
        Legacy method - keeping for compatibility but not used in new implementation.
        """
        pass
    
    def convert_coordinates(self, landmark, frame_width, frame_height):
        """
        Returns MediaPipe's normalized coordinates (0.0-1.0) directly.
        Both X and Y coordinates are already normalized between 0 and 1.
        
        🔧 TO CHANGE COORDINATE SYSTEM:
        - MediaPipe already gives normalized coordinates (0.0-1.0)
        - X: 0.0=left edge, 1.0=right edge of camera view
        - Y: 0.0=top edge, 1.0=bottom edge of camera view
        
        Args:
            landmark: MediaPipe landmark object
            frame_width (int): Width of the camera frame (used for display only)
            frame_height (int): Height of the camera frame (used for display only)
            
        Returns:
            list: [x, y] coordinates normalized between 0.0 and 1.0
        """
        # MediaPipe gives us coordinates from 0.0 to 1.0 - perfect!
        # We keep them normalized instead of converting to pixels
        x = float(landmark.x)    # x position normalized (0.0-1.0)
        y = float(landmark.y)    # y position normalized (0.0-1.0)
        
        return [x, y]
    
    def draw_tracking_info(self, frame, coords, name):
        """
        Draws visual feedback on the camera feed.
        Converts normalized coordinates back to pixels for display.
        
        🔧 TO CHANGE VISUAL DISPLAY:
        - Modify colors, sizes, or text here
        - Add more visual elements if desired
        
        Args:
            frame: The camera frame to draw on
            coords (list): [x, y] normalized coordinates (0.0-1.0)
            name (str): Name of the landmark
        """
        # Convert normalized coordinates back to pixel coordinates for display
        frame_height, frame_width = frame.shape[:2]
        x = int(coords[0] * frame_width)   # Convert 0.0-1.0 back to pixels
        y = int(coords[1] * frame_height)  # Convert 0.0-1.0 back to pixels
        
        # Draw a green circle at the tracked point
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Add text label showing both pixel and normalized coordinates
        label = f"{name.replace('_', ' ').title()}"
        norm_label = f"({coords[0]:.3f}, {coords[1]:.3f})"
        
        cv2.putText(frame, label, (x + 10, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, norm_label, (x + 10, y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def process_frame(self, frame, send_osc=True):
        """
        Main processing function - analyzes one camera frame and sends OSC data.
        
        🔧 TO CHANGE WHAT DATA GETS SENT:
        - Modify the OSC address format in send_arm_data() method
        - Change what coordinates get sent
        - Add calculations for angles, distances, etc.
        
        Args:
            frame: Camera frame to process
            send_osc (bool): Whether to send OSC messages this frame
            
        Returns:
            frame: Processed frame with visual overlays
        """
        # Convert camera image from BGR (Blue-Green-Red) to RGB (Red-Green-Blue)
        # MediaPipe expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run pose detection on the frame
        results = self.pose.process(rgb_frame)
        
        # Check if we found a person in the frame
        if results.pose_landmarks:
            # Draw the full skeleton on the frame (optional visual feedback)
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Get all the detected landmarks (body points)
            landmarks = results.pose_landmarks.landmark
            
            # ============ COLLECT ARM DATA ============
            # We need to collect data for each arm separately
            arm_data = {'left': {}, 'right': {}}
            
            # Process each body part we're interested in
            for landmark_name, landmark_index in self.landmarks.items():
                
                # Make sure this landmark exists in the detected pose
                if landmark_index < len(landmarks):
                    landmark = landmarks[landmark_index]
                    
                    # Convert to normalized coordinates (0.0-1.0)
                    coords = self.convert_coordinates(landmark, frame_width, frame_height)
                    
                    # Apply smoothing to reduce jitter
                    smoothed_coords = self.smooth_coordinates(landmark_name, coords)
                    
                    # Store the data by arm side and joint type
                    if 'left' in landmark_name:
                        joint_type = landmark_name.replace('left_', '')
                        arm_data['left'][joint_type] = smoothed_coords
                    else:
                        joint_type = landmark_name.replace('right_', '')
                        arm_data['right'][joint_type] = smoothed_coords
                    
                    # Draw visual feedback on the camera feed
                    self.draw_tracking_info(frame, smoothed_coords, landmark_name)
            
            # ============ SEND ARM DATA VIA OSC ============
            # Only send OSC messages when send_osc is True (every 3rd frame)
            if send_osc:
                # Send complete arm data if we have all three points for each arm
                
                # Check and send left arm data
                if all(joint in arm_data['left'] for joint in ['shoulder', 'elbow', 'wrist']):
                    self.send_arm_data('left', 
                                     arm_data['left']['shoulder'],
                                     arm_data['left']['elbow'], 
                                     arm_data['left']['wrist'])
                
                # Check and send right arm data  
                if all(joint in arm_data['right'] for joint in ['shoulder', 'elbow', 'wrist']):
                    self.send_arm_data('right',
                                      arm_data['right']['shoulder'],
                                      arm_data['right']['elbow'],
                                      arm_data['right']['wrist'])
        
        return frame
    
    def setup_camera(self):
        """
        Initialize and configure the camera.
        
        🔧 TO CHANGE CAMERA SETTINGS:
        - Modify camera_width, camera_height, camera_fps in __init__
        - Change the camera index (0) if you have multiple cameras
        
        Returns:
            cv2.VideoCapture: Configured camera object, or None if failed
        """
        # Try to open the default camera (index 0)
        # 🔧 CHANGE CAMERA: Use 1, 2, etc. for other cameras
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            print("💡 Try changing the camera index in setup_camera() method")
            return None
        
        # Configure camera settings for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        
        print(f"📹 Camera initialized: {self.camera_width}x{self.camera_height} @ {self.camera_fps}fps")
        return cap
    
    def print_startup_info(self):
        """Prints helpful information when the program starts."""
        print("\n" + "="*50)
        print("🎯 ARM TRACKING ACTIVE!")
        print("="*50)
        print("\n📤 OSC Messages being sent:")
        print("   /arm/left  [shoulderX, shoulderY, elbowX, elbowY, wristX, wristY] -> port 8001")
        print("   /arm/right [shoulderX, shoulderY, elbowX, elbowY, wristX, wristY] -> port 8002")
        print("   (All coordinates normalized 0.0-1.0)")
        
        print(f"\n🎛️  Controls:")
        print("   'q' = Quit the program")
        print("   's' = Toggle smoothing level")
        print(f"\n⚙️  Current Settings:")
        print(f"   Smoothing: {self.smoothing_factor}")
        print(f"   Camera: {self.camera_width}x{self.camera_height}")
        print("="*50 + "\n")
    
    def update_display_info(self, frame, fps):
        """
        Adds informational text to the camera display.
        
        🔧 TO CHANGE DISPLAY INFO:
        - Modify the text, colors, or positions here
        - Add more information if desired
        
        Args:
            frame: Camera frame to draw on
            fps (float): Current frames per second
        """
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Current smoothing level
        cv2.putText(frame, f"Smoothing: {self.smoothing_factor:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def handle_keyboard_input(self, key):
        """
        Handles keyboard controls during tracking.
        
        🔧 TO ADD MORE CONTROLS:
        - Add more elif statements with different key codes
        - Use ord('letter') to check for specific keys
        
        Args:
            key: The pressed key code
            
        Returns:
            bool: True if program should continue, False if should quit
        """
        if key == ord('q'):
            print("👋 Quitting...")
            return False
            
        elif key == ord('s'):
            # Toggle between high and low smoothing
            if self.smoothing_factor < 0.8:
                self.smoothing_factor = 0.9  # High smoothing
                print("🔄 Smoothing: HIGH (0.9) - Smoother but slower response")
            else:
                self.smoothing_factor = 0.3  # Low smoothing  
                print("🔄 Smoothing: LOW (0.3) - Faster but more jittery")
                
        # 🔧 ADD MORE CONTROLS HERE:
        # elif key == ord('r'):  # Reset something
        #     print("Reset!")
        # elif key == ord('c'):  # Change something
        #     print("Changed!")
            
        return True  # Continue running
    
    def run(self):
        """
        Main tracking loop - this is where everything happens!
        
        🔧 TO CHANGE THE MAIN BEHAVIOR:
        - Modify the frame processing logic
        - Add additional calculations or data processing
        - Change the display window properties
        """
        # ============ CAMERA INITIALIZATION ============
        cap = self.setup_camera()
        if cap is None:
            return  # Exit if camera failed to initialize
        
        # ============ STARTUP INFO ============
        self.print_startup_info()
        
        # ============ PERFORMANCE TRACKING ============
        frame_count = 0
        start_time = time.time()
        current_fps = 0
        
        # ============ OSC THROTTLING ============
        # 🔧 CHANGE THIS to send OSC messages at different rates:
        osc_frame_counter = 0
        send_every_n_frames = 1  # Send OSC every N frames (reduces from ~30/sec to ~30/N/sec)
        
        try:
            # ============ MAIN TRACKING LOOP ============
            # This loop runs continuously until you press 'q'
            while True:
                # Read one frame from the camera
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading camera frame")
                    break
                
                # ============ FRAME PREPROCESSING ============
                # Flip the frame horizontally so it works like a mirror
                # 🔧 REMOVE THIS LINE if you don't want mirror mode:
                frame = cv2.flip(frame, 1)
                
                # ============ POSE DETECTION & OSC SENDING ============
                # This is where the magic happens!
                
                # Increment OSC frame counter
                osc_frame_counter += 1
                
                # Only send OSC messages every N frames to reduce network traffic
                send_osc = (osc_frame_counter % send_every_n_frames == 0)
                
                frame = self.process_frame(frame, send_osc)
                
                # ============ FPS CALCULATION ============
                # Calculate frames per second every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = 30 / elapsed
                    start_time = time.time()
                
                # ============ DISPLAY UPDATES ============
                # Add text and info to the camera display
                self.update_display_info(frame, current_fps)
                
                # Show the camera feed with tracking overlays
                # 🔧 CHANGE WINDOW NAME here if desired:
                cv2.imshow('MediaPipe Arm Tracker - Press Q to Quit', frame)
                
                # ============ KEYBOARD INPUT HANDLING ============
                # Check if any keys were pressed (wait 1 millisecond)
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break  # User pressed 'q' to quit
                
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n⚠️  Interrupted by user (Ctrl+C)")
        
        finally:
            # ============ CLEANUP ============
            # Always clean up resources when done
            cap.release()           # Release the camera
            cv2.destroyAllWindows() # Close all windows
            print("✅ Tracking stopped - cleanup complete")


def main():
    """
    Main function - START HERE to customize your settings!
    
    🔧 MAIN SETTINGS TO CHANGE:
    """
    
    # ============ OSC CONFIGURATION ============
    # 🔧 CHANGE THESE to send OSC data to different programs:
    OSC_IP = "10.0.0.4"     # IP address of the receiving program
    
    # Ports are now fixed: 8001 for left arm, 8002 for right arm
    
    # ============ STARTUP MESSAGES ============
    print("\n" + "🎵" * 20)
    print("   MEDIAPIPE ARM TRACKER WITH OSC")
    print("🎵" * 20)
    print(f"\n🎯 OSC Target: {OSC_IP}")
    print("   Left arm  -> port 8001")
    print("   Right arm -> port 8002")
    print("📋 Make sure your OSC receiver is running!")
    print("\n💡 Common OSC receivers:")
    print("   - Max/MSP, Pure Data, SuperCollider (music)")
    print("   - TouchDesigner, Processing (visuals)")
    print("   - Unity, Unreal Engine (games)")
    print("   - VRChat OSC, VTube Studio (avatars)")
    
    # Create and run tracker (ports are fixed at 8001 and 8002)
    tracker = ArmTracker(OSC_IP, 0)  # Port parameter not used anymore
    tracker.run()


# ============ CUSTOMIZATION GUIDE ============
"""
🔧 COMMON THINGS YOU MIGHT WANT TO CHANGE:

1. TRACK DIFFERENT BODY PARTS:
   - Go to __init__ method, find self.landmarks dictionary
   - Add/remove entries. MediaPipe pose landmark numbers:
     * 0=nose, 7=left_ear, 8=right_ear
     * 23=left_hip, 24=right_hip
     * 25=left_knee, 26=right_knee, 27=left_ankle, 28=right_ankle

2. CHANGE OSC MESSAGE FORMAT:
   - Go to process_frame() method
   - Modify the osc_address variable
   - Change what data gets sent in smoothed_coords

3. ADJUST CAMERA SETTINGS:
   - Go to __init__ method
   - Change camera_width, camera_height, camera_fps

4. MODIFY SMOOTHING:
   - Go to __init__ method
   - Change smoothing_factor (0.0-0.9)

5. ADD ANGLE CALCULATIONS:
   - Add methods to calculate angles between joints
   - Send angle data via OSC instead of/in addition to coordinates

6. CHANGE COORDINATE SCALING:
   - Go to convert_coordinates() method
   - Modify the scaling calculations

7. ADD MORE KEYBOARD CONTROLS:
   - Go to handle_keyboard_input() method
   - Add more elif statements for different keys
"""

if __name__ == "__main__":
    main()