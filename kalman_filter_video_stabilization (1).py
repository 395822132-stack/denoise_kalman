import os
import cv2
import numpy as np
from tqdm import tqdm
import time

class KalmanFilter:
    def __init__(self, state_dim=4, measure_dim=2):
        self.kf = cv2.KalmanFilter(state_dim, measure_dim)
        
        # State transition matrix
        self.kf.transitionMatrix = np.eye(state_dim, dtype=np.float32)
        for i in range(state_dim - measure_dim):
            self.kf.transitionMatrix[i, i + measure_dim] = 1.0
        
        # Measurement matrix
        self.kf.measurementMatrix = np.zeros((measure_dim, state_dim), dtype=np.float32)
        for i in range(measure_dim):
            self.kf.measurementMatrix[i, i] = 1.0
        
        # Process noise covariance
        self.kf.processNoiseCov = 1e-4 * np.eye(state_dim, dtype=np.float32)
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = 1e-2 * np.eye(measure_dim, dtype=np.float32)
        
        # Posterior error covariance
        self.kf.errorCovPost = np.eye(state_dim, dtype=np.float32)
        
        # Initial state
        self.kf.statePost = np.zeros((state_dim, 1), dtype=np.float32)
        
        self.first_measurement = True
    
    def update(self, measurement):
        if self.first_measurement:
            self.kf.statePost[:2] = measurement.reshape(2, 1)
            self.first_measurement = False
        
        # Prediction
        prediction = self.kf.predict()
        
        # Update
        measured = np.array(measurement, dtype=np.float32).reshape(2, 1)
        estimated = self.kf.correct(measured)
        
        return prediction, estimated

class VideoStabilizer:
    def __init__(self):
        self.kf = KalmanFilter()
        self.prev_frame = None
        self.prev_points = None
        self.prev_gray = None
        self.orb = cv2.ORB_create()
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    def detect_features(self, gray):
        """Detect feature points, using multiple methods to ensure sufficient feature points"""
        # Method 1: Use ORB
        keypoints = self.orb.detect(gray, None)
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        
        # If ORB detects insufficient points, use GoodFeaturesToTrack
        if len(points) < 50:
            points = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
            if points is not None:
                points = points.reshape(-1, 2)
            else:
                points = np.array([], dtype=np.float32).reshape(0, 2)
        
        return points
    
    def stabilize(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect feature points
            self.prev_points = self.detect_features(self.prev_gray)
            
            # If no feature points are detected, return the original frame
            if len(self.prev_points) == 0:
                return frame
            
            return frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        try:
            curr_points, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params
            )
        except cv2.error as e:
            print(f"Optical flow calculation error: {e}")
            # Re-detect feature points
            self.prev_points = self.detect_features(gray)
            self.prev_gray = gray.copy()
            self.prev_frame = frame.copy()
            return frame
        
        # Filter valid points
        idx = np.where(status == 1)[0]
        if len(idx) < 4:
            # If there are not enough valid points, re-detect feature points
            self.prev_points = self.detect_features(gray)
            self.prev_gray = gray.copy()
            self.prev_frame = frame.copy()
            return frame
        
        prev_pts = self.prev_points[idx]
        curr_pts = curr_points[idx]
        
        # Calculate global motion (translation)
        dx = np.median(curr_pts[:, 0] - prev_pts[:, 0])
        dy = np.median(curr_pts[:, 1] - prev_pts[:, 1])
        
        # Smooth motion using Kalman filter
        measurement = np.array([dx, dy])
        prediction, estimated = self.kf.update(measurement)
        
        # Get smoothed motion vector
        smooth_dx = estimated[0][0]
        smooth_dy = estimated[1][0]
        
        # Build affine transformation matrix
        H = np.array([[1, 0, -smooth_dx],
                      [0, 1, -smooth_dy]], dtype=np.float32)
        
        # Apply transformation
        height, width = frame.shape[:2]
        stabilized = cv2.warpAffine(frame, H, (width, height))
        
        # Update previous frame information
        self.prev_frame = stabilized
        self.prev_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        
        # Detect new feature points
        self.prev_points = self.detect_features(self.prev_gray)
        
        return stabilized

def process_videos(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create performance statistics file
    performance_file = os.path.join(output_dir, "performance_stats.txt")
    with open(performance_file, "w") as f:
        f.write("Video Processing Performance Statistics\n")
        f.write("=" * 50 + "\n")
    
    # Iterate through all classes
    for class_name in ['001', '002', '003', '004', '005']:
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        
        if not os.path.exists(class_input_dir):
            print(f"Skipping non-existent directory: {class_input_dir}")
            continue
        
        os.makedirs(class_output_dir, exist_ok=True)
        
        print(f"Processing class {class_name}...")
        
        # Get all video files
        video_files = [f for f in os.listdir(class_input_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        for video_file in video_files:
            input_path = os.path.join(class_input_dir, video_file)
            output_path = os.path.join(class_output_dir, video_file)
            
            print(f"Processing video: {video_file}")
            
            # Initialize video stabilizer
            stabilizer = VideoStabilizer()
            
            # Open video file
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                print(f"Cannot open video file: {input_path}")
                continue
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process each frame and time it
            processed_frames = 0
            start_time = time.time()
            
            pbar = tqdm(total=total_frames, desc=f"Processing {video_file}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Apply video stabilization
                    stabilized_frame = stabilizer.stabilize(frame)
                    
                    # Write the processed frame
                    out.write(stabilized_frame)
                    processed_frames += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    # Write the original frame as a fallback
                    out.write(frame)
                    processed_frames += 1
                    pbar.update(1)
            
            pbar.close()
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            if processing_time > 0:
                processing_fps = processed_frames / processing_time
            else:
                processing_fps = 0
            
            # Output performance statistics
            print(f"Video '{video_file}' processing completed:")
            print(f"  Total frames: {processed_frames}")
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Processing speed: {processing_fps:.2f} FPS")
            if fps > 0:
                print(f"  Real-time ratio: {processing_fps/fps:.2f}x (relative to original video)")
            
            # Save performance statistics to file
            with open(performance_file, "a") as f:
                f.write(f"\nVideo: {class_name}/{video_file}\n")
                f.write(f"  Total frames: {processed_frames}\n")
                f.write(f"  Processing time: {processing_time:.2f} seconds\n")
                f.write(f"  Processing speed: {processing_fps:.2f} FPS\n")
                if fps > 0:
                    f.write(f"  Real-time ratio: {processing_fps/fps:.2f}x (relative to original video)\n")
            
            # Release resources
            cap.release()
            out.release()

def main():
    # Set input and output directories
    train_input_dir = "./rocket_dataset/train/003/b19ec5431d387119cb4ba8985c3f4ae6_brightness_L2_part002_aug0008.mp4"
    #test_input_dir = "./rocket_dataset/test"
    train_output_dir = "./rocket_dataset_stabilized/train"
    #test_output_dir = "./rocket_dataset_stabilized/test"
    
    # Process training set
    print("Starting to process training set...")
    process_videos(train_input_dir, train_output_dir)
    
    # Process test set
    print("Starting to process test set...")
    #process_videos(test_input_dir, test_output_dir)
    
    print("All video processing completed!")
    print("Performance statistics have been saved to the performance_stats.txt file in the output directory")

if __name__ == "__main__":
    main()