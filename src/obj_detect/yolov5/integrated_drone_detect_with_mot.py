#!/usr/bin/env python3
"""
Advanced Drone Person Tracking with Multi-Object Tracking (MOT) + Person Re-Identification
Uses YOLOv5 for detection + ByteTrack-style tracking for persistent ID tracking + Person Re-ID

KEY IMPROVEMENTS:
1. Detects people with YOLOv5
2. Tracks them across frames with persistent IDs using ByteTrack algorithm
3. Locks onto ONE target and follows it even through occlusions
4. **NEW**: Extracts appearance features for person re-identification
5. **NEW**: Hybrid approach - visual servoing to 3m, then blind navigation with re-ID

This prevents:
- Losing target when detection flickers
- Switching between different people
- Confusion from multiple detections
- Re-centering loop during final approach
- Mistaking similar people for the target

Based on research paper: "Autonomous Vision-Based Mobile Target Tracking"
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import numpy as np
from pathlib import Path
import threading
import time
from collections import deque
import scipy.optimize as opt
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F

# DroneKit imports
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import math

# YOLOv5 imports
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding box in image space
    State: [center_x, center_y, width, height, vel_x, vel_y, vel_w, vel_h]
    """
    count = 0
    
    def __init__(self, bbox):
        """Initialize tracker with first detection [x1, y1, x2, y2, confidence]"""
        # Extract bbox
        x1, y1, x2, y2 = bbox[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        # State: [cx, cy, w, h, vcx, vcy, vw, vh]
        self.kf = np.eye(8)
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0]).reshape(8, 1)
        
        # Process noise
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 10  # Higher noise for velocities
        
        # Measurement noise
        self.R = np.eye(4) * 10
        
        # Measurement matrix (we observe cx, cy, w, h)
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)
        
        # State transition (simple constant velocity model)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)  # position += velocity
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        
        # Store confidence
        self.confidence = bbox[4] if len(bbox) > 4 else 1.0
        
    def predict(self):
        """Advance state by one timestep"""
        # Predict: x = F*x
        self.state = self.F @ self.state
        
        # Update covariance: P = F*P*F' + Q
        # (simplified, no actual P matrix for speed)
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self.get_bbox()
    
    def update(self, bbox):
        """Update state with new detection"""
        x1, y1, x2, y2 = bbox[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        measurement = np.array([cx, cy, w, h]).reshape(4, 1)
        
        # Kalman update: x = x + K*(z - H*x)
        innovation = measurement - self.H @ self.state
        self.state[:4] += 0.5 * innovation  # Simplified gain
        
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # Update confidence
        if len(bbox) > 4:
            self.confidence = 0.7 * self.confidence + 0.3 * bbox[4]
    
    def get_bbox(self):
        """Return current bbox as [x1, y1, x2, y2, confidence]"""
        cx, cy, w, h = self.state[:4, 0]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return np.array([x1, y1, x2, y2, self.confidence])


class ByteTracker:
    """
    ByteTrack-style Multi-Object Tracker
    
    Key features:
    - Associates detections with tracks using IoU matching
    - Handles low-confidence detections for robustness
    - Maintains tracks through brief occlusions
    - Assigns persistent IDs to each tracked object
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Args:
            max_age: Maximum frames to keep alive track without detections
            min_hits: Minimum hits before track is confirmed
            iou_threshold: Minimum IoU for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: numpy array of detections [[x1, y1, x2, y2, conf], ...]
            
        Returns:
            List of active tracks [[x1, y1, x2, y2, track_id], ...]
        """
        self.frame_count += 1
        
        # Get predictions from existing trackers
        trks = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks.append(pos)
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = [trk for i, trk in enumerate(trks) if i not in to_del]
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            self.trackers.append(trk)
        
        # Return confirmed tracks
        ret = []
        for trk in self.trackers:
            # Only return tracks that have been seen enough times
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_bbox()
                ret.append(np.concatenate((d[:4], [trk.id])).reshape(1, -1))
        
        # Remove dead tracks
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    @staticmethod
    def iou(bb_test, bb_gt):
        """Calculate IoU between two bboxes [x1, y1, x2, y2]"""
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        intersection = w * h
        area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        union = area_test + area_gt - intersection
        return intersection / union if union > 0 else 0
    
    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Match detections to trackers using Hungarian algorithm
        
        Returns:
            matched: List of [det_idx, trk_idx] pairs
            unmatched_detections: List of detection indices
            unmatched_trackers: List of tracker indices
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det[:4], trk[:4])
        
        # Hungarian algorithm for optimal assignment
        if iou_matrix.max() > iou_threshold:
            # Use linear_sum_assignment from scipy
            matched_indices = np.array(list(zip(*opt.linear_sum_assignment(-iou_matrix))))
            
            # Filter out matches with low IoU
            matches = []
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] < iou_threshold:
                    continue
                matches.append(m.reshape(1, 2))
            
            if len(matches) == 0:
                matches = np.empty((0, 2), dtype=int)
            else:
                matches = np.concatenate(matches, axis=0)
            
            unmatched_detections = [d for d in range(len(detections)) if d not in matches[:, 0]]
            unmatched_trackers = [t for t in range(len(trackers)) if t not in matches[:, 1]]
            
            return matches, unmatched_detections, unmatched_trackers
        else:
            return [], list(range(len(detections))), list(range(len(trackers)))


class PersonReIDExtractor:
    """
    Person Re-Identification feature extractor using ResNet50
    Extracts appearance features from person bounding boxes for re-identification
    """
    def __init__(self, device='cuda'):
        """Initialize Re-ID model with pre-trained ResNet50"""
        rospy.loginfo("🔧 Initializing Person Re-ID model...")
        
        self.device = device
        
        # Use ResNet50 pre-trained on ImageNet as feature extractor
        # Remove final classification layer to get 2048-dim feature vector
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove FC layer
        self.model.eval()
        self.model.to(device)
        
        # Image preprocessing for ResNet
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),  # Standard person Re-ID size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        rospy.loginfo("✅ Person Re-ID model loaded (ResNet50)")
        rospy.loginfo("   Feature dimension: 2048")
    
    def extract_features(self, image, bbox):
        """
        Extract appearance features from a person bounding box
        
        Args:
            image: Full frame (numpy array, BGR)
            bbox: [x1, y1, x2, y2] bounding box coordinates
        
        Returns:
            features: 2048-dim normalized feature vector (numpy array)
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Ensure valid bbox
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Crop person from image
        person_img = image[y1:y2, x1:x2]
        
        if person_img.size == 0:
            rospy.logwarn("⚠️ Empty bbox for Re-ID feature extraction")
            return np.zeros(2048)
        
        # Convert BGR to RGB
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess and extract features
        with torch.no_grad():
            img_tensor = self.preprocess(person_img).unsqueeze(0).to(self.device)
            features = self.model(img_tensor)
            features = features.squeeze().cpu().numpy()
            
            # L2 normalization for cosine similarity
            features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    @staticmethod
    def compute_similarity(features1, features2):
        """
        Compute cosine similarity between two feature vectors
        
        Returns:
            similarity: Float in [0, 1], higher = more similar
        """
        if features1 is None or features2 is None:
            return 0.0
        
        # Cosine similarity = dot product of normalized vectors
        similarity = np.dot(features1, features2)
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]


class DroneController:
    """Drone controller with MOT-enhanced tracking + Person Re-ID for robust following"""
    
    def __init__(self, connection_string='udp:127.0.0.1:14550'):
        print("Connecting to vehicle...")
        self.vehicle = connect(connection_string, wait_ready=True, timeout=60)
        
        # State management
        self.state = "IDLE"
        self.target_lock = threading.Lock()
        
        # MOT: Track ONE person with persistent ID
        self.locked_track_id = None  # Once we lock, we follow THIS ID
        self.track_lost_count = 0  # How many frames without our locked track
        self.max_track_lost_frames = 60  # Give up after 60 frames (~2 seconds at 30fps)
        
        # Centering parameters - RELAXED FOR REALISM
        self.centering_threshold_pixels = 60  # ±60px tolerance (increased for stability)
        self.centering_hysteresis = 80  # ±80px tolerance once centered (prevents timer reset)
        
        # Approach control
        self.approach_enabled = True  # Enable approach after centering
        
        # PID control parameters for YAW (horizontal) - GENTLE TUNING
        self.pid_yaw_kp = 0.02  # Much lower - prevents huge swings
        self.pid_yaw_ki = 0.0005  # Lower - builds up slowly
        self.pid_yaw_kd = 0.02  # REDUCED from 0.04 - less overshoot
        self.pid_yaw_error_sum = 0
        self.pid_yaw_last_error = 0
        
        # PID control parameters for ALTITUDE (vertical) - GENTLE TUNING
        self.pid_alt_kp = 0.0005  # REDUCED from 0.0010 - MUCH GENTLER (was causing runaway)
        self.pid_alt_ki = 0.00001  # REDUCED from 0.00003 - slower buildup
        self.pid_alt_kd = 0.0010  # REDUCED from 0.0015 - minimal overshoot
        self.pid_alt_error_sum = 0
        self.pid_alt_last_error = 0
        
        # Limits - REDUCED FOR SMOOTHER MOTION
        self.max_yaw_step = 3.0  # Reduced from 8.0 - smaller rotations
        self.max_altitude_step = 0.15  # CRITICAL FIX: Reduced from 0.3m - was causing runaway up
        self.min_yaw_step = 0.3  # Min rotation (avoid jitter)
        self.min_altitude_step = 0.05  # Min altitude change
        
        # CRITICAL: Altitude safety limits to prevent flyaway
        self.max_altitude_absolute = 12.0  # Never exceed 12m altitude
        self.min_altitude_absolute = 8.0   # Never go below 8m (stay above person)
        
        # Timing - SLOWER FOR SMOOTH MOVEMENT
        self.last_adjustment_time = 0
        self.base_adjustment_delay = 0.6  # Increased from 0.3 - gives time to settle
        self.centered_hold_time = 5  # REDUCED from 10s - start approaching sooner
        
        # Tracking
        self.centered_start_time = None
        self.detection_lost_timeout = 30.0  # INCREASED from 15s - more patience
        self.last_detection_time = 0
        
        # Recovery search when target lost
        self.recovery_search_enabled = True
        self.recovery_search_angle = 4
        self.recovery_search_altitude = 2
        self.recovery_search_speed = 5
        self.last_known_heading = None
        self.last_known_altitude = None
        self.in_recovery_search = False
        
        # Deadband - LARGER to prevent small movements triggering adjustments
        self.deadband_pixels_yaw = 5  # Horizontal deadband (was 3)
        self.deadband_pixels_alt = 30  # CRITICAL: Larger vertical deadband to reduce altitude hunting
        
        # Movement history for derivative calculation
        self.x_offset_history = deque(maxlen=10)
        self.y_offset_history = deque(maxlen=10)
        self.timestamp_history = deque(maxlen=10)
        
        # Velocity estimation
        self.x_velocity = 0
        self.y_velocity = 0
        
        # Distance estimation (monocular depth from bbox)
        self.person_real_height = 1.7  # Average person height in meters
        self.camera_focal_length = 640  # Approx focal length in pixels
        self.target_approach_distance = 5.0  # USER REQUIREMENT: Stop at 5 meters from target
        self.approach_mode = True  # Enable diagonal approach after centering
        self.min_approach_distance = 0.5  # Minimum safe distance (50cm)
        self.max_approach_distance = 15.0  # Maximum detection range
        self.approach_tolerance = 80  # Allow ±80px offset during approach
        
        # State tracking
        self.estimated_distance = None
        self.approach_complete = False
        self.hovering_complete = False  # NEW: Track 10-second hover completion
        self.hover_start_time = None    # NEW: When hovering started
        self.hover_duration = 10.0      # NEW: Hover for 10 seconds at target
        self.rtl_complete = False       # NEW: Return-to-launch completion
        self.initial_position = None    # NEW: Save takeoff position for RTL
        self.last_approach_time = 0
        self.approach_delay = 2.5  # Delay between approach movements
        
        # ===== PRECISE TRAJECTORY PLANNING =====
        # User requirement: "precisely calculate how much down and forward we want to go and do as fast movement as possible"
        self.trajectory_calculated = False  # Have we calculated the approach trajectory?
        self.trajectory_forward_m = None    # Meters to move forward
        self.trajectory_vertical_m = None   # Meters to move up/down
        self.trajectory_lateral_m = None    # Meters to move left/right
        self.trajectory_executing = False   # Currently executing calculated trajectory
        self.trajectory_start_time = None   # When trajectory execution started
        self.trajectory_start_position = None  # Position when trajectory started
        rospy.loginfo("🎯 Precise Trajectory Mode: Calculate exact movement, execute smoothly, no jitter!")
        
        # ===== PERSON RE-ID FOR HYBRID APPROACH =====
        self.reid_extractor = None  # Will be initialized with device info
        self.target_features = None  # Saved appearance features of locked target
        self.reid_threshold = 0.25  # Similarity threshold for re-identification (VERY LOW - only reject if completely different person)
        self.hybrid_approach_enabled = True  # Enable hybrid approach mode
        self.hybrid_transition_distance = 3.0  # Switch to blind navigation at 3 meters
        self.in_blind_approach = False  # Flag for blind navigation phase
        rospy.loginfo("🎯 Hybrid Approach Mode: Visual servoing to 3m, then blind navigation with Re-ID")
        self.approach_delay = 2.5  # INCREASED from 1.5s - give GPS more time to complete movement
        
        # CRITICAL: Track cumulative altitude changes to detect runaway
        self.altitude_at_lock = None  # Altitude when target was locked
        self.cumulative_altitude_change = 0.0  # Total altitude change since lock
        self.max_cumulative_altitude_change = 1.5  # Max 1.5m total altitude adjustment
        
        # NON-BLOCKING forward/backward movement tracking
        self.forward_movement_active = False  # Flag for ongoing movement
        self.forward_movement_start_time = None  # When movement started
        self.forward_movement_duration = 0  # How long to move
        self.forward_movement_msg = None  # Velocity command to send
        
        # Derivative rate limiting
        self.max_derivative_yaw = 100  # px/s
        self.max_derivative_alt = 50   # px/s
        
        print(f"✅ Drone connected: {self.vehicle.version}")
        print("🎯 MOT Tracking Mode: Will lock onto ONE person and follow persistently")
    
    def arm_and_takeoff(self, target_altitude):
        print("Arming motors...")
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True

        while not self.vehicle.armed:
            print(" Waiting for arming...")
            time.sleep(1)

        print("Taking off!")
        self.vehicle.simple_takeoff(target_altitude)

        while True:
            alt = self.vehicle.location.global_relative_frame.alt
            print(f" Altitude: {alt:.2f} m")
            if alt >= target_altitude * 0.95:
                print("Reached target altitude")
                # Save initial position for Return-To-Launch
                self.initial_position = {
                    'lat': self.vehicle.location.global_relative_frame.lat,
                    'lon': self.vehicle.location.global_relative_frame.lon,
                    'alt': self.vehicle.location.global_relative_frame.alt
                }
                rospy.loginfo(f"📍 INITIAL POSITION SAVED: Lat={self.initial_position['lat']:.7f}, Lon={self.initial_position['lon']:.7f}, Alt={self.initial_position['alt']:.2f}m")
                break
            time.sleep(1)
    
    def condition_yaw(self, heading, relative=False, clockwise=True, speed=10):
        """Rotate drone"""
        is_relative = 1 if relative else 0
        direction = 1 if clockwise else -1
        
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            heading, speed, direction, is_relative,
            0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
    
    def adjust_altitude(self, delta_meters):
        """Adjust altitude by delta_meters (positive = up, negative = down)"""
        current_location = self.vehicle.location.global_relative_frame
        current_alt = current_location.alt
        
        # Track cumulative altitude changes to detect runaway
        if self.altitude_at_lock is None:
            self.altitude_at_lock = current_alt
        
        # Check if cumulative change would exceed limit
        potential_cumulative = self.cumulative_altitude_change + delta_meters
        if abs(potential_cumulative) > self.max_cumulative_altitude_change:
            rospy.logwarn(f"🚫 CUMULATIVE ALTITUDE LIMIT! Already changed {self.cumulative_altitude_change:.2f}m since lock. Capping further adjustments.")
            # Allow only partial adjustment to reach limit
            remaining_budget = self.max_cumulative_altitude_change - abs(self.cumulative_altitude_change)
            delta_meters = remaining_budget * (1 if delta_meters > 0 else -1)
            if abs(delta_meters) < 0.01:  # Less than 1cm
                rospy.logwarn("  → No altitude budget remaining, skipping adjustment")
                return
        
        # Calculate new altitude with safety limits
        new_altitude = current_alt + delta_meters
        
        # CRITICAL: Enforce absolute altitude limits to prevent flyaway
        if new_altitude > self.max_altitude_absolute:
            rospy.logwarn(f"🚫 ALTITUDE LIMIT! Capping at {self.max_altitude_absolute}m (tried {new_altitude:.2f}m)")
            new_altitude = self.max_altitude_absolute
        elif new_altitude < self.min_altitude_absolute:
            rospy.logwarn(f"🚫 ALTITUDE FLOOR! Raising to {self.min_altitude_absolute}m (tried {new_altitude:.2f}m)")
            new_altitude = self.min_altitude_absolute
        
        # Also enforce minimum safe altitude (2m above ground)
        new_altitude = max(2.0, new_altitude)
        
        # Update cumulative tracking
        actual_delta = new_altitude - current_alt
        self.cumulative_altitude_change += actual_delta
        
        target_location = LocationGlobalRelative(
            current_location.lat,
            current_location.lon,
            new_altitude
        )
        
        rospy.loginfo(f"Adjusting altitude: {current_alt:.2f}m → {new_altitude:.2f}m (cumulative: {self.cumulative_altitude_change:+.2f}m)")
        self.vehicle.simple_goto(target_location)
    
    def move_forward_backward(self, distance_meters):
        """
        START a forward/backward movement (NON-BLOCKING)
        Movement continues in background, check update_forward_movement() for completion
        """
        # Get current heading to move in that direction
        current_heading = self.vehicle.heading  # 0-359 degrees
        heading_rad = math.radians(current_heading)
        
        # Calculate velocity components (NED frame)
        velocity_mps = 0.3 if abs(distance_meters) >= 0.3 else 0.2  # 0.3 m/s or 0.2 m/s
        
        # Calculate North and East components based on heading
        velocity_north = velocity_mps * math.cos(heading_rad) * (1 if distance_meters > 0 else -1)
        velocity_east = velocity_mps * math.sin(heading_rad) * (1 if distance_meters > 0 else -1)
        
        # Duration: how long to move to cover the distance
        duration = abs(distance_meters) / velocity_mps
        duration = max(0.3, min(1.0, duration))  # Clamp between 0.3-1.0 seconds (SHORTER!)
        
        rospy.loginfo(f"  → Starting {'FORWARD' if distance_meters > 0 else 'BACKWARD'} {abs(distance_meters):.2f}m at {velocity_mps}m/s for {duration:.1f}s")
        rospy.loginfo(f"  → Velocity: N={velocity_north:.2f} E={velocity_east:.2f} (heading={current_heading}°)")
        
        # Create velocity command (NED frame: North, East, Down)
        self.forward_movement_msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,       # time_boot_ms (not used)
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
            0b0000111111000111,  # type_mask (use velocity)
            0, 0, 0,  # x, y, z positions (not used)
            velocity_north, velocity_east, 0,  # vx, vy, vz velocity
            0, 0, 0,  # afx, afy, afz acceleration (not used)
            0, 0)     # yaw, yaw_rate (not used)
        
        # Start the movement (non-blocking)
        self.forward_movement_active = True
        self.forward_movement_start_time = time.time()
        self.forward_movement_duration = duration
        
        # Send first command
        self.vehicle.send_mavlink(self.forward_movement_msg)
        self.vehicle.flush()
    
    def update_forward_movement(self):
        """
        Update ongoing forward/backward movement (called every frame)
        Returns True if movement is still ongoing, False if complete
        """
        if not self.forward_movement_active:
            return False
        
        elapsed = time.time() - self.forward_movement_start_time
        
        if elapsed >= self.forward_movement_duration:
            # Movement complete - STOP
            stop_msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,
                0, 0, 0,
                0, 0, 0,  # Zero velocity
                0, 0, 0,
                0, 0)
            self.vehicle.send_mavlink(stop_msg)
            self.vehicle.flush()
            
            self.forward_movement_active = False
            self.forward_movement_msg = None
            rospy.loginfo(f"  ✓ Movement complete")
            return False
        else:
            # Continue movement - resend velocity command
            self.vehicle.send_mavlink(self.forward_movement_msg)
            self.vehicle.flush()
            return True
    
    def return_to_launch(self):
        """
        Return to initial takeoff position instantly using GUIDED mode
        User requirement: "go back to the initial position instantly"
        """
        if self.initial_position is None:
            rospy.logwarn("⚠️ No initial position saved - cannot RTL!")
            return False
        
        rospy.logwarn("=" * 70)
        rospy.logwarn("🏠 RETURNING TO LAUNCH POSITION...")
        rospy.logwarn(f"   Target: Lat={self.initial_position['lat']:.7f}, Lon={self.initial_position['lon']:.7f}, Alt={self.initial_position['alt']:.2f}m")
        rospy.logwarn("=" * 70)
        
        # Ensure GUIDED mode
        self.vehicle.mode = VehicleMode("GUIDED")
        
        # Create target location
        target_location = LocationGlobalRelative(
            self.initial_position['lat'],
            self.initial_position['lon'],
            self.initial_position['alt']
        )
        
        # Command drone to go to initial position
        self.vehicle.simple_goto(target_location)
        
        # Wait until we're close to the initial position
        while True:
            current_location = self.vehicle.location.global_relative_frame
            
            # Calculate distance to target
            dlat = current_location.lat - self.initial_position['lat']
            dlon = current_location.lon - self.initial_position['lon']
            distance = math.sqrt((dlat * 111320) ** 2 + (dlon * 111320 * math.cos(math.radians(current_location.lat))) ** 2)
            alt_error = abs(current_location.alt - self.initial_position['alt'])
            
            rospy.loginfo(f"   Returning... Horizontal distance: {distance:.2f}m, Altitude error: {alt_error:.2f}m")
            
            # Check if we're close enough (within 1 meter horizontally and 0.5m vertically)
            if distance < 1.0 and alt_error < 0.5:
                rospy.logwarn("✓ REACHED INITIAL POSITION!")
                self.rtl_complete = True
                break
            
            time.sleep(0.5)
            
            if rospy.is_shutdown():
                break
        
        return True
    
    def calculate_precise_trajectory(self, x_offset_px, y_offset_px, image_width, image_height, bbox_height):
        """
        PRECISE TRAJECTORY CALCULATION
        User requirement: "precisely calculate how much down and forward we want to go"
        
        Calculate EXACT movements needed based on current visual information:
        1. Forward distance: Close gap from current distance to target (0.7m)
        2. Vertical offset: Height adjustment based on Y pixel error
        3. Lateral offset: Left/right adjustment based on X pixel error
        
        Returns: (forward_m, lateral_m, vertical_m) in meters
        """
        if self.estimated_distance is None:
            rospy.logwarn("⚠️ Cannot calculate trajectory - no distance estimate!")
            return None, None, None
        
        # 1. FORWARD MOVEMENT: Close the distance gap
        forward_distance_needed = self.estimated_distance - self.target_approach_distance
        forward_distance_needed = max(0, forward_distance_needed)  # Don't go backward
        
        # 2. LATERAL MOVEMENT: Convert pixel offset to real-world meters
        # Using pinhole camera model: real_offset = (pixel_offset * distance) / focal_length
        lateral_offset_m = (x_offset_px * self.estimated_distance) / self.camera_focal_length
        lateral_offset_m = max(-2.0, min(2.0, lateral_offset_m))  # Cap at ±2m
        
        # 3. VERTICAL MOVEMENT: Convert pixel offset to real-world meters
        vertical_offset_m = (y_offset_px * self.estimated_distance) / self.camera_focal_length
        vertical_offset_m = max(-1.5, min(1.5, vertical_offset_m))  # Cap at ±1.5m
        
        rospy.logwarn("=" * 70)
        rospy.logwarn("📐 PRECISE TRAJECTORY CALCULATED:")
        rospy.logwarn(f"   Current distance: {self.estimated_distance:.2f}m → Target: {self.target_approach_distance:.2f}m")
        rospy.logwarn(f"   🔹 FORWARD: {forward_distance_needed:.2f}m")
        rospy.logwarn(f"   🔹 LATERAL: {lateral_offset_m:+.2f}m ({'←LEFT' if lateral_offset_m < 0 else '→RIGHT'})")
        rospy.logwarn(f"   🔹 VERTICAL: {vertical_offset_m:+.2f}m ({'↑UP' if vertical_offset_m < 0 else '↓DOWN'})")
        rospy.logwarn("=" * 70)
        
        return forward_distance_needed, lateral_offset_m, vertical_offset_m
    
    def check_edge_danger_during_trajectory(self, bbox, image_width, image_height):
        """
        Check if target bbox is dangerously close to frame edges
        Used during trajectory execution to detect emergency situations
        
        Returns: (has_danger, emergency_velocities)
            has_danger: True if immediate correction needed
            emergency_velocities: (lateral, vertical) in m/s (body frame), None if no danger
        """
        if bbox is None:
            return False, None
        
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Calculate bbox edges
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        bbox_left = target_center_x - bbox_width / 2
        bbox_right = target_center_x + bbox_width / 2
        bbox_top = target_center_y - bbox_height / 2
        bbox_bottom = target_center_y + bbox_height / 2
        
        # Edge padding (same as approach mode)
        edge_padding_horizontal = 150
        edge_padding_vertical = 150
        
        # Check dangers
        danger_left = bbox_left < edge_padding_horizontal
        danger_right = bbox_right > (image_width - edge_padding_horizontal)
        danger_top = bbox_top < edge_padding_vertical
        danger_bottom = bbox_bottom > (image_height - edge_padding_vertical)
        
        if not (danger_left or danger_right or danger_top or danger_bottom):
            return False, None
        
        # Calculate emergency corrections (body frame)
        emergency_lateral = 0.0
        emergency_vertical = 0.0
        
        if danger_left:
            rospy.logwarn(f"🚨 TRAJECTORY INTERRUPT: LEFT edge (bbox_left={bbox_left:.0f}px)")
            emergency_lateral = +0.40  # Move RIGHT
        elif danger_right:
            rospy.logwarn(f"🚨 TRAJECTORY INTERRUPT: RIGHT edge (bbox_right={bbox_right:.0f}px)")
            emergency_lateral = -0.40  # Move LEFT
        
        if danger_top:
            rospy.logwarn(f"🚨 TRAJECTORY INTERRUPT: TOP edge (bbox_top={bbox_top:.0f}px)")
            emergency_vertical = -0.35  # ASCEND
        elif danger_bottom:
            rospy.logwarn(f"🚨 TRAJECTORY INTERRUPT: BOTTOM edge (bbox_bottom={bbox_bottom:.0f}px)")
            emergency_vertical = +0.35  # DESCEND
        
        return True, (emergency_lateral, emergency_vertical)
    
    def execute_precise_trajectory(self, bbox_tracker=None, image_width=None, image_height=None):
        """
        HYBRID CALIBRATED SYSTEM: Trajectory Planning + Emergency Override
        User requirement: "both working hand in hand calibrated"
        
        PRIMARY MODE: Smooth trajectory execution (0.5 m/s coordinated movement)
        SAFETY NET: Emergency edge monitoring (interrupts with 0.35-0.40 m/s corrections)
        
        Args:
            bbox_tracker: Function that returns current bbox [x1, y1, x2, y2] or None
            image_width, image_height: Frame dimensions for edge detection
        
        Process:
        1. Execute smooth trajectory as planned
        2. Monitor edges continuously (10 Hz)
        3. If danger detected → PAUSE trajectory, apply emergency correction
        4. After correction → RECALCULATE remaining trajectory, resume
        5. Continues even if vision lost (blind completion)
        """
        if self.trajectory_forward_m is None:
            rospy.logwarn("⚠️ No trajectory calculated!")
            return False
        
        # Calculate total distance to cover
        total_distance = math.sqrt(
            self.trajectory_forward_m**2 + 
            self.trajectory_lateral_m**2 + 
            self.trajectory_vertical_m**2
        )
        
        if total_distance < 0.1:  # Less than 10cm - already there!
            rospy.loginfo("✓ Already at target position!")
            return True
        
        # Calculate movement duration (fast but safe)
        # velocity = 0.5 m/s for smooth approach
        movement_velocity = 0.5  # m/s
        movement_duration = total_distance / movement_velocity
        movement_duration = max(1.0, min(10.0, movement_duration))  # 1-10 seconds
        
        # Calculate velocity components (all axes move proportionally)
        forward_velocity = self.trajectory_forward_m / movement_duration
        lateral_velocity_body = self.trajectory_lateral_m / movement_duration
        vertical_velocity = self.trajectory_vertical_m / movement_duration  # NED frame
        
        rospy.logwarn("🚀 HYBRID SYSTEM: EXECUTING PRECISE TRAJECTORY WITH EDGE MONITORING")
        rospy.logwarn(f"   Duration: {movement_duration:.1f}s @ {movement_velocity:.2f}m/s")
        rospy.logwarn(f"   Velocities: Fwd={forward_velocity:.2f} Lat={lateral_velocity_body:+.2f} Vert={vertical_velocity:+.2f} m/s")
        if bbox_tracker is not None:
            rospy.logwarn(f"   🛡️ Emergency Override: ACTIVE (monitoring edges)")
        else:
            rospy.logwarn(f"   🛡️ Emergency Override: DISABLED (blind mode)")
        
        # Execute trajectory
        self.trajectory_executing = True
        self.trajectory_start_time = time.time()
        self.trajectory_start_position = self.vehicle.location.global_relative_frame
        
        start_time = time.time()
        emergency_pause_time = 0.0  # Track time spent in emergency corrections
        
        while True:
            elapsed = time.time() - start_time - emergency_pause_time
            
            if elapsed >= movement_duration:
                # Trajectory complete - STOP
                rospy.logwarn("✓ TRAJECTORY EXECUTION COMPLETE!")
                stop_msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
                    0, 0, 0,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    0b0000111111000111,
                    0, 0, 0,
                    0, 0, 0,  # Zero velocity
                    0, 0, 0,
                    0, 0)
                self.vehicle.send_mavlink(stop_msg)
                self.vehicle.flush()
                self.trajectory_executing = False
                return True
            
            # ===== HYBRID SYSTEM: Check for edge danger =====
            has_danger = False
            if bbox_tracker is not None and image_width is not None and image_height is not None:
                current_bbox = bbox_tracker()
                if current_bbox is not None:
                    has_danger, emergency_velocities = self.check_edge_danger_during_trajectory(
                        current_bbox, image_width, image_height
                    )
                    
                    if has_danger:
                        # ===== EMERGENCY INTERRUPT: Pause trajectory, apply correction =====
                        rospy.logwarn("⚠️⚠️⚠️ EMERGENCY OVERRIDE ACTIVE - PAUSING TRAJECTORY")
                        emergency_start = time.time()
                        
                        emergency_lateral, emergency_vertical = emergency_velocities
                        
                        # Get current heading for NED conversion
                        current_heading = self.vehicle.heading
                        heading_rad = math.radians(current_heading)
                        
                        # Convert emergency corrections to NED
                        velocity_north_emerg = emergency_lateral * math.cos(heading_rad + math.pi/2)
                        velocity_east_emerg = emergency_lateral * math.sin(heading_rad + math.pi/2)
                        velocity_down_emerg = emergency_vertical
                        
                        rospy.logwarn(f"   Emergency velocities: Lat={emergency_lateral:+.2f} Vert={emergency_vertical:+.2f} m/s")
                        
                        # Apply emergency correction for 0.5 seconds
                        emergency_duration = 0.5
                        emergency_elapsed = 0.0
                        
                        while emergency_elapsed < emergency_duration:
                            emergency_msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
                                0, 0, 0,
                                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                                0b0000111111000111,
                                0, 0, 0,
                                velocity_north_emerg, velocity_east_emerg, velocity_down_emerg,
                                0, 0, 0,
                                0, 0)
                            self.vehicle.send_mavlink(emergency_msg)
                            self.vehicle.flush()
                            
                            time.sleep(0.1)
                            emergency_elapsed += 0.1
                        
                        # Track emergency pause time (doesn't count toward trajectory completion)
                        emergency_pause_time += time.time() - emergency_start
                        
                        rospy.logwarn("   ✓ Emergency correction applied, resuming trajectory...")
                        continue  # Re-check immediately after correction
            
            # ===== NORMAL TRAJECTORY EXECUTION =====
            # Get current heading for NED conversion
            current_heading = self.vehicle.heading
            heading_rad = math.radians(current_heading)
            
            # Convert body-frame velocities to NED
            velocity_north = (forward_velocity * math.cos(heading_rad) + 
                             lateral_velocity_body * math.cos(heading_rad + math.pi/2))
            velocity_east = (forward_velocity * math.sin(heading_rad) + 
                            lateral_velocity_body * math.sin(heading_rad + math.pi/2))
            velocity_down = vertical_velocity
            
            # Send trajectory velocity command
            msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,
                0, 0, 0,
                velocity_north, velocity_east, velocity_down,
                0, 0, 0,
                0, 0)
            self.vehicle.send_mavlink(msg)
            self.vehicle.flush()
            
            # Log progress every second
            if int(elapsed) != int(elapsed - 0.1):
                progress = (elapsed / movement_duration) * 100
                rospy.loginfo(f"   Progress: {progress:.0f}% ({elapsed:.1f}/{movement_duration:.1f}s)")
            
            time.sleep(0.1)  # 10 Hz update rate
            
            if rospy.is_shutdown():
                break
        
        return False
    
    def estimate_distance(self, bbox_height_pixels, image_height):
        """Estimate distance to person using monocular depth estimation"""
        if bbox_height_pixels <= 0:
            return None
        
        distance = (self.person_real_height * self.camera_focal_length) / bbox_height_pixels
        distance = max(self.min_approach_distance, min(self.max_approach_distance, distance))
        
        self.estimated_distance = distance
        return distance
    
    def approach_target(self, x_offset_px, y_offset_px, image_width, image_height, bbox_width=None, bbox_height=None):
        """
        HYBRID DIAGONAL APPROACH: 
        - Moves forward while SIMULTANEOUSLY correcting X/Y offsets
        - Phase 1 (>3m): Visual servoing with centering
        - Phase 2 (<3m): Blind navigation with Re-ID verification
        
        Args:
            x_offset_px: Horizontal offset from center (pixels)
            y_offset_px: Vertical offset from center (pixels)
            image_width: Frame width (pixels)
            image_height: Frame height (pixels)
            bbox_width: Width of bounding box (pixels) - used for edge detection
            bbox_height: Height of bounding box (pixels) - used for edge detection
        """
        if self.estimated_distance is None:
            rospy.logwarn("⚠️ No distance estimate - cannot approach!")
            rospy.logwarn("   Make sure bbox_height is being passed to center_target()")
            return False
        
        rospy.loginfo(f"🎯 APPROACH: Current distance: {self.estimated_distance:.1f}m, Target: {self.target_approach_distance:.1f}m")
        
        distance_error = self.estimated_distance - self.target_approach_distance
        
        if abs(distance_error) < 0.8:  # Within 80cm of target (wider tolerance for 5m distance)
            rospy.loginfo(f"✓ At target distance: {self.estimated_distance:.1f}m (target: {self.target_approach_distance:.1f}m)")
            if not self.approach_complete:
                # Just reached target - start hovering!
                self.approach_complete = True
                self.hover_start_time = time.time()
                rospy.logwarn("=" * 70)
                rospy.logwarn(f"🎯 APPROACH COMPLETE! At {self.estimated_distance:.1f}m from target")
                rospy.logwarn("   Starting 10-second hover with ZERO VELOCITY (not ALT_HOLD)...")
                rospy.logwarn("=" * 70)
            return True
        
        # ===== HYBRID APPROACH LOGIC =====
        # Check if we should transition to blind navigation
        if self.hybrid_approach_enabled and self.estimated_distance <= self.hybrid_transition_distance:
            if not self.in_blind_approach:
                rospy.logwarn("=" * 60)
                rospy.logwarn(f"🚀 HYBRID TRANSITION: Switching to BLIND NAVIGATION mode!")
                rospy.logwarn(f"   Distance: {self.estimated_distance:.1f}m <= {self.hybrid_transition_distance:.1f}m")
                rospy.logwarn("   No more centering - using Re-ID for verification")
                rospy.logwarn("=" * 60)
                self.in_blind_approach = True
        
        if distance_error > 0:  # Too far - approach
            # Calculate forward speed based on distance (faster when far, slower when close)
            if distance_error > 5.0:
                forward_velocity = 0.5  # 0.5 m/s when far away
            elif distance_error > 2.0:
                forward_velocity = 0.3  # 0.3 m/s at medium range
            else:
                forward_velocity = 0.2  # 0.2 m/s when getting close
            
            # ===== SMART APPROACH STRATEGY =====
            # Goal: Get as CLOSE as possible while keeping target visible
            # Philosophy: 
            #   - Target should FILL the frame (bigger = closer = good!)
            #   - Only correct offsets if target is LEAVING frame (emergency)
            #   - Prioritize FORWARD movement to get close
            
            # ===== EDGE DANGER DETECTION: Check BBOX EDGES, not center! =====
            # User requirement: 100-150px padding BEFORE bbox touches actual frame edge
            edge_padding_horizontal = 150  # Padding from left/right edges
            edge_padding_vertical = 150    # Padding from top/bottom edges
            
            # Calculate target CENTER position in frame
            target_center_x = (image_width / 2) + x_offset_px
            target_center_y = (image_height / 2) + y_offset_px
            
            # Calculate BBOX EDGES (not just center!)
            # This ensures we detect danger BEFORE bbox touches actual frame edges
            if bbox_width is not None and bbox_height is not None:
                bbox_left = target_center_x - (bbox_width / 2)
                bbox_right = target_center_x + (bbox_width / 2)
                bbox_top = target_center_y - (bbox_height / 2)
                bbox_bottom = target_center_y + (bbox_height / 2)
            else:
                # Fallback: assume average person bbox size (100x200px)
                rospy.logwarn("⚠️ No bbox dimensions provided - using estimated size")
                bbox_left = target_center_x - 50
                bbox_right = target_center_x + 50
                bbox_top = target_center_y - 100
                bbox_bottom = target_center_y + 100
            
            # Check if BBOX EDGES are within padding distance from frame edges
            danger_left = bbox_left < edge_padding_horizontal
            danger_right = bbox_right > (image_width - edge_padding_horizontal)
            danger_top = bbox_top < edge_padding_vertical
            danger_bottom = bbox_bottom > (image_height - edge_padding_vertical)
            
            in_danger_zone = danger_left or danger_right or danger_top or danger_bottom
            
            # ===== ADAPTIVE CORRECTION: Gentle normally, AGGRESSIVE OVERRIDE in danger =====
            
            # ===== EMERGENCY EDGE OVERRIDE - NON-AVOIDABLE! =====
            # User requirement: "make the edge robust and not avoidable, if the target is about to touch them mitigate immediately overriding other command"
            # When target is dangerously close to edges, STOP FORWARD MOVEMENT and AGGRESSIVELY correct!
            
            emergency_override = False
            override_forward = 0.0
            override_lateral = 0.0
            override_vertical = 0.0
            
            if in_danger_zone:
                # EMERGENCY MODE: OVERRIDE ALL NORMAL COMMANDS!
                emergency_override = True
                lateral_gain = 0.0040   # 8x normal (VERY AGGRESSIVE!)
                vertical_gain = 0.0030  # 8x normal (VERY AGGRESSIVE!)
                
                # STOP or SLOW forward movement during emergency
                forward_scale = 0.2     # Reduce to 20% - prioritize correction over approach!
                
                if danger_left:
                    rospy.logwarn(f"🚨🚨 EMERGENCY: LEFT edge (bbox_left={bbox_left:.0f}px < padding={edge_padding_horizontal}px)")
                    # Force HARD RIGHT correction
                    override_lateral = +0.40  # Move RIGHT at 40cm/s!
                    
                if danger_right:
                    rospy.logwarn(f"🚨🚨 EMERGENCY: RIGHT edge (bbox_right={bbox_right:.0f}px > frame={image_width-edge_padding_horizontal}px)")
                    # Force HARD LEFT correction
                    override_lateral = -0.40  # Move LEFT at 40cm/s!
                    
                if danger_top:
                    rospy.logwarn(f"🚨🚨 EMERGENCY: TOP edge (bbox_top={bbox_top:.0f}px < padding={edge_padding_vertical}px)")
                    # BUG FIX: Target too HIGH → need to ASCEND (negative in NED)
                    override_vertical = -0.35  # ASCEND at 35cm/s! (was +0.35 WRONG!)
                    override_lateral = 0.0     # NO lateral during vertical emergency!
                    
                if danger_bottom:
                    rospy.logwarn(f"🚨🚨 EMERGENCY: BOTTOM edge (bbox_bottom={bbox_bottom:.0f}px > frame={image_height-edge_padding_vertical}px)")
                    # Target too LOW → need to DESCEND (positive in NED)
                    override_vertical = +0.35  # DESCEND at 35cm/s!
                    override_lateral = 0.0     # NO lateral during vertical emergency!
                    
            else:
                # NORMAL MODE: Gentle corrections, prioritize forward movement
                lateral_gain = 0.0003   # Gentle lateral correction
                vertical_gain = 0.0002  # Gentle vertical correction (let target grow in frame!)
                forward_scale = 1.0     # Full forward speed (prioritize getting close)
                
                rospy.loginfo(f"   Bbox edges: left={bbox_left:.0f}px right={bbox_right:.0f}px top={bbox_top:.0f}px bottom={bbox_bottom:.0f}px (safe zone)")
            
            # Apply forward speed scaling
            forward_velocity = forward_velocity * forward_scale
            
            # ===== COMPUTE VELOCITY CORRECTIONS =====
            
            if emergency_override:
                # EMERGENCY OVERRIDE: Use pre-calculated emergency velocities!
                lateral_correction = override_lateral
                vertical_correction = override_vertical
                rospy.logwarn(f"   ⚠️⚠️ EMERGENCY OVERRIDE ACTIVE: Lat={lateral_correction:+.2f} Vert={vertical_correction:+.2f} m/s")
            else:
                # NORMAL CORRECTIONS: Calculate from pixel offsets
                
                # Priority-based corrections: disable lateral when vertical edges trigger
                if danger_top or danger_bottom:
                    # Vertical edge danger - ONLY forward + vertical, NO lateral!
                    lateral_correction = 0.0
                    rospy.loginfo("   ⚠️ Vertical edge danger - lateral corrections DISABLED")
                else:
                    # Safe vertically - allow lateral corrections
                    lateral_correction = -x_offset_px * lateral_gain
                    lateral_correction = max(-0.30, min(0.30, lateral_correction))
                
                # Vertical correction (up/down) - NED FRAME!
                # NED: Positive DOWN = descend, Negative DOWN = ascend
                # Image coords: Y < 0 (target HIGH/TOP) → should ASCEND (negative DOWN)
                #               Y > 0 (target LOW/BOTTOM) → should DESCEND (positive DOWN)
                # Therefore: DOWN_velocity = +y_offset * gain (POSITIVE sign!)
                vertical_correction = +y_offset_px * vertical_gain
                vertical_correction = max(-0.25, min(0.25, vertical_correction))
            
            # Calculate velocity in NED frame
            current_heading = self.vehicle.heading
            heading_rad = math.radians(current_heading)
            
            # Forward velocity (along heading)
            velocity_north_fwd = forward_velocity * math.cos(heading_rad)
            velocity_east_fwd = forward_velocity * math.sin(heading_rad)
            
            # Lateral velocity (perpendicular to heading)
            # Perpendicular is heading + 90°
            velocity_north_lat = lateral_correction * math.cos(heading_rad + math.pi/2)
            velocity_east_lat = lateral_correction * math.sin(heading_rad + math.pi/2)
            
            # Total velocity (diagonal movement!)
            velocity_north = velocity_north_fwd + velocity_north_lat
            velocity_east = velocity_east_fwd + velocity_east_lat
            velocity_down = vertical_correction
            
            # Log approach mode
            if self.in_blind_approach:
                rospy.logwarn(f"➡️ BLIND DIAGONAL APPROACH: Fwd={forward_velocity:.2f} Lat={lateral_correction:+.2f} Vert={vertical_correction:+.2f} m/s")
            else:
                rospy.logwarn(f"➡️ VISUAL DIAGONAL APPROACH: Fwd={forward_velocity:.2f} Lat={lateral_correction:+.2f} Vert={vertical_correction:+.2f} m/s")
            
            rospy.loginfo(f"   Offsets: X={x_offset_px:+d}px Y={y_offset_px:+d}px | Distance: {self.estimated_distance:.1f}m")
            rospy.loginfo(f"   🚀 NED Velocity: N={velocity_north:.2f} E={velocity_east:.2f} D={velocity_down:.2f} m/s")
            
            # Send velocity command using MAVLink (NON-BLOCKING - just send once, approach() manages timing)
            msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
                0,       # time_boot_ms (not used)
                0, 0,    # target system, target component
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
                0b0000111111000111,  # type_mask (only velocity enabled)
                0, 0, 0,  # x, y, z positions (not used)
                velocity_north, velocity_east, velocity_down,  # velocities in m/s
                0, 0, 0,  # x, y, z acceleration (not used)
                0, 0)     # yaw, yaw_rate (not used)
            
            # Send command (non-blocking)
            self.vehicle.send_mavlink(msg)
            self.vehicle.flush()
            
            rospy.loginfo(f"   ✓ Diagonal approach command sent")
            
        else:  # Too close
            rospy.logwarn(f"⬅️ Too close! ({self.estimated_distance:.1f}m < {self.target_approach_distance:.1f}m)")
        
        return False
    
    def center_target(self, x_offset, y_offset, image_width, image_height, bbox_height=None, bbox_width=None):
        """Center target using PID control with adaptive gains"""
        current_time = time.time()
        
        # ===== UPDATE NON-BLOCKING FORWARD MOVEMENT =====
        movement_ongoing = self.update_forward_movement()
        if movement_ongoing:
            # Movement in progress - skip adjustments but continue tracking
            return False
        
        # ===== SKIP CENTERING IF ALREADY APPROACHING =====
        if self.state == "APPROACHING":
            # Already in approach mode - CONTINUOUS visual servoing!
            # Send velocity commands every frame to keep target centered
            if self.approach_mode and not self.approach_complete:
                # Update distance estimate
                if bbox_height is not None:
                    distance = self.estimate_distance(bbox_height, image_height)
                    if distance:
                        # Only log occasionally to avoid spam
                        if int(current_time * 2) % 10 == 0:  # Every 5 seconds
                            rospy.loginfo(f"� Distance: {distance:.1f}m")
                
                # CONTINUOUS approach - send velocity commands every frame!
                # This is TRUE visual servoing (like centering phase)
                time_since_last_approach = current_time - self.last_approach_time
                
                # Only log "DIAGONAL APPROACH" occasionally to avoid spam
                if time_since_last_approach >= 2.5:
                    rospy.logwarn(f"� DIAGONAL APPROACH! Distance: {self.estimated_distance:.1f}m → {self.target_approach_distance:.1f}m")
                    self.last_approach_time = current_time
                
                # Call approach function with current offsets AND bbox dimensions
                # This runs EVERY FRAME, not every 2.5 seconds!
                approach_result = self.approach_target(x_offset, y_offset, image_width, image_height, bbox_width, bbox_height)
                
                if approach_result:
                    rospy.loginfo("🎯 APPROACH COMPLETE! At target distance.")
                    self.state = "LOCKED"
                    return True
                else:
                    return False
            else:
                # Approach already complete
                self.state = "LOCKED"
                return True
        
        # ===== HYBRID APPROACH: Skip centering during blind navigation =====
        if self.in_blind_approach:
            rospy.loginfo("🚀 BLIND NAVIGATION: Skipping centering, using Re-ID only")
            # Still update distance estimate
            if bbox_height is not None:
                distance = self.estimate_distance(bbox_height, image_height)
                if distance:
                    rospy.loginfo(f"📏 Distance: {distance:.1f}m (blind mode)")
            return False  # Don't center, just approach
        
        # Add current measurements
        self.x_offset_history.append(x_offset)
        self.y_offset_history.append(y_offset)
        self.timestamp_history.append(current_time)
        
        # Calculate velocity (rate of change)
        if len(self.timestamp_history) >= 3:
            dt = self.timestamp_history[-1] - self.timestamp_history[-3]
            if dt > 0:
                self.x_velocity = (self.x_offset_history[-1] - self.x_offset_history[-3]) / dt
                self.y_velocity = (self.y_offset_history[-1] - self.y_offset_history[-3]) / dt
        
        # Adaptive rate limiting
        distance_2d = math.sqrt(x_offset**2 + y_offset**2)
        
        if distance_2d > 100:
            adjustment_delay = self.base_adjustment_delay
        elif distance_2d > 50:
            adjustment_delay = self.base_adjustment_delay * 1.2
        else:
            adjustment_delay = self.base_adjustment_delay * 1.5
        
        if current_time - self.last_adjustment_time < adjustment_delay:
            return False
        
        dt = current_time - self.last_adjustment_time if self.last_adjustment_time > 0 else 0.3
        
        # During approach, use wider tolerance
        active_tolerance = self.approach_tolerance if (self.approach_mode and self.centered_start_time) else self.centering_threshold_pixels
        
        # Use hysteresis: once centered, allow more wobble before resetting timer
        if self.centered_start_time is not None:
            # Already centered - use wider tolerance to prevent timer reset
            check_tolerance = self.centering_hysteresis
        else:
            # Not yet centered - use normal tolerance
            check_tolerance = active_tolerance
        
        # Check if centered
        if abs(x_offset) < check_tolerance and abs(y_offset) < check_tolerance:
            
            # Reset PID integral terms when centered
            self.pid_yaw_error_sum = 0
            self.pid_alt_error_sum = 0
            
            if self.centered_start_time is None:
                self.centered_start_time = current_time
                rospy.logwarn("🎯 TARGET CENTERED! Holding position...")
            
            hold_duration = current_time - self.centered_start_time
            
            # Estimate distance while centered
            if bbox_height is not None:
                distance = self.estimate_distance(bbox_height, image_height)
                if distance:
                    rospy.loginfo(f"📏 Estimated distance: {distance:.1f}m (bbox_height={bbox_height}px)")
                else:
                    rospy.logwarn(f"⚠️ Distance estimation failed! bbox_height={bbox_height}px")
            else:
                rospy.logwarn("⚠️ No bbox_height provided - cannot estimate distance!")
            
            if hold_duration >= self.centered_hold_time:
                # After locking, enter approach mode
                rospy.logwarn(f"🔓 HOLD COMPLETE! ({hold_duration:.1f}s) - Starting approach phase")
                rospy.loginfo(f"   approach_mode={self.approach_mode}, complete={self.approach_complete}, distance={self.estimated_distance}")
                
                # CRITICAL: Change state to APPROACHING so we don't keep calling center_target()
                if self.state == "CENTERING":
                    self.state = "APPROACHING"
                    rospy.logwarn("🚁 STATE TRANSITION: CENTERING → APPROACHING")
                
                if self.approach_mode and not self.approach_complete:
                    # Log approach start only once
                    time_since_last_approach = current_time - self.last_approach_time
                    if time_since_last_approach >= 2.5:  # Only for logging
                        rospy.logwarn(f"🚁 APPROACHING TARGET! Distance: {self.estimated_distance:.1f}m → {self.target_approach_distance:.1f}m")
                        self.last_approach_time = current_time
                    
                    # Call approach function CONTINUOUSLY (every frame)
                    approach_result = self.approach_target(x_offset, y_offset, image_width, image_height, bbox_width, bbox_height)
                    
                    if approach_result:
                        rospy.loginfo("🎯 APPROACH COMPLETE! At target distance.")
                        self.state = "LOCKED"
                        return True
                    else:
                        return False
                else:
                    if not self.approach_mode:
                        rospy.logwarn("⚠️ APPROACH DISABLED - locked without approaching")
                    if self.approach_complete:
                        rospy.loginfo("✓ APPROACH ALREADY COMPLETE")
                    
                    rospy.loginfo(f"✓✓✓ LOCKED! Held for {hold_duration:.1f}s ✓✓✓")
                    self.state = "LOCKED"
                    return True
            else:
                remaining = self.centered_hold_time - hold_duration
                rospy.loginfo(f"Holding... {remaining:.1f}s remaining")
                return False
        else:
            # Not centered - reset timer if it was previously set
            if self.centered_start_time is not None:
                rospy.logwarn(f"⚠️ LOST CENTER! Offset: X={x_offset:+d}px Y={y_offset:+d}px (tolerance: ±{check_tolerance}px) - Resetting hold timer")
            self.centered_start_time = None
        
        # Calculate offset percentages
        x_percent = (x_offset / image_width) * 100
        y_percent = (y_offset / image_height) * 100
        
        rospy.loginfo(f"Offset: X={x_offset:+4d}px ({x_percent:+.1f}%) Y={y_offset:+4d}px ({y_percent:+.1f}%) | 2D: {distance_2d:.1f}px | Vel: X={self.x_velocity:+.1f} Y={self.y_velocity:+.1f}")
        
        adjustment_made = False
        
        # ========== INTELLIGENT VERTICAL CENTERING (Y-AXIS) ==========
        # Smart strategy: Use FORWARD/BACK movement primarily, altitude as backup
        # - Target TOO LOW (negative Y) → MOVE FORWARD (brings target UP in frame)
        # - Target TOO HIGH (positive Y) → MOVE BACK or DOWN
        
        if abs(y_offset) > self.deadband_pixels_alt:  # Use larger deadband for altitude
            error = y_offset
            
            # Decide: Use forward/back movement OR altitude adjustment
            # Forward/back is MORE EFFECTIVE for vertical centering!
            
            if error < -50:  # Target TOO LOW (more than 50px below center)
                # MOVE FORWARD to bring target UP in frame (BALANCED distance)
                forward_distance = 0.20  # INCREASED from 0.15m - 20cm per adjustment (faster centering)
                rospy.logwarn(f"📐 Target TOO LOW ({error}px) → MOVING FORWARD {forward_distance}m to center")
                self.move_forward_backward(forward_distance)
                adjustment_made = True
                
            elif error > 50:  # Target TOO HIGH (more than 50px above center)
                # MOVE BACKWARD to bring target DOWN in frame
                backward_distance = -0.20  # INCREASED from -0.15m - 20cm per adjustment
                rospy.logwarn(f"📐 Target TOO HIGH ({error}px) → MOVING BACKWARD {abs(backward_distance)}m to center")
                self.move_forward_backward(backward_distance)
                adjustment_made = True
                
            else:
                # Small vertical offset: Use traditional altitude adjustment
                p_term = self.pid_alt_kp * error
                
                self.pid_alt_error_sum += error * dt
                self.pid_alt_error_sum = max(-500, min(500, self.pid_alt_error_sum))
                i_term = self.pid_alt_ki * self.pid_alt_error_sum
                
                error_rate = (error - self.pid_alt_last_error) / dt if dt > 0 else 0
                error_rate = max(-50, min(50, error_rate))
                d_term = self.pid_alt_kd * error_rate
                
                altitude_step = -(p_term + i_term + d_term)
                altitude_step = max(-self.max_altitude_step, min(self.max_altitude_step, altitude_step))
                
                if abs(altitude_step) < self.min_altitude_step:
                    altitude_step = self.min_altitude_step * (1 if altitude_step > 0 else -1)
                
                self.pid_alt_last_error = error
                
                if altitude_step < 0:
                    rospy.logwarn(f"⬇️  DOWN {abs(altitude_step):.3f}m [P:{p_term:.3f} I:{i_term:.3f} D:{d_term:.3f}]")
                else:
                    rospy.logwarn(f"⬆️  UP {abs(altitude_step):.3f}m [P:{p_term:.3f} I:{i_term:.3f} D:{d_term:.3f}]")
                
                self.adjust_altitude(altitude_step)
                adjustment_made = True
        else:
            self.pid_alt_error_sum = 0
            self.pid_alt_last_error = 0
            rospy.loginfo(f"  Vertical centered: {abs(y_offset)}px < {self.deadband_pixels_alt}px")
        
        # ========== YAW PID CONTROL (Horizontal) ==========
        if abs(x_offset) > self.deadband_pixels_yaw:  # Use separate deadband for yaw
            error = x_offset
            
            p_term = self.pid_yaw_kp * error
            
            self.pid_yaw_error_sum += error * dt
            self.pid_yaw_error_sum = max(-500, min(500, self.pid_yaw_error_sum))
            i_term = self.pid_yaw_ki * self.pid_yaw_error_sum
            
            error_rate = (error - self.pid_yaw_last_error) / dt if dt > 0 else 0
            error_rate = max(-100, min(100, error_rate))
            d_term = self.pid_yaw_kd * error_rate
            
            yaw_step = p_term + i_term + d_term
            yaw_step = max(-self.max_yaw_step, min(self.max_yaw_step, yaw_step))
            
            if abs(yaw_step) < self.min_yaw_step:
                yaw_step = self.min_yaw_step * (1 if yaw_step > 0 else -1)
            
            self.pid_yaw_last_error = error
            
            current_heading = self.vehicle.heading
            new_heading = (current_heading + yaw_step) % 360
            
            if yaw_step > 0:
                rospy.logwarn(f"➡️  RIGHT {yaw_step:.2f}°: {current_heading:.0f}°→{new_heading:.0f}° [P:{p_term:.2f} I:{i_term:.2f} D:{d_term:.2f}]")
                self.condition_yaw(new_heading, relative=False, clockwise=True, speed=15)
            else:
                rospy.logwarn(f"⬅️  LEFT {abs(yaw_step):.2f}°: {current_heading:.0f}°→{new_heading:.0f}° [P:{p_term:.2f} I:{i_term:.2f} D:{d_term:.2f}]")
                self.condition_yaw(new_heading, relative=False, clockwise=False, speed=15)
            
            adjustment_made = True
        else:
            self.pid_yaw_error_sum = 0
            self.pid_yaw_last_error = 0
        
        if adjustment_made:
            self.last_adjustment_time = current_time
        
        return False
    
    def rotate_and_search(self, rotation_steps=24, pause_time=3):
        """Slow rotation search"""
        print(f"Starting search: {rotation_steps} positions, {pause_time}s pause")
        degrees_per_step = 360 / rotation_steps
        
        for i in range(rotation_steps):
            # Check if we've locked onto a track
            if self.locked_track_id is not None:
                print(f"🔒 Locked onto track ID {self.locked_track_id} during search! Stopping rotation...")
                return True
            
            if self.state in ["CENTERING", "LOCKED"]:
                print("🎯 Target detected! Stopping search...")
                return True
            
            target_heading = i * degrees_per_step
            print(f"Rotating to {target_heading:.0f}° ({i+1}/{rotation_steps})...")
            
            self.condition_yaw(target_heading, relative=False, clockwise=True, speed=5)
            time.sleep(degrees_per_step / 5.0 + 1.0)
            
            print(f"  Scanning at {target_heading:.0f}°...")
            for scan_step in range(int(pause_time * 2)):
                # Check for confirmed track lock
                if self.locked_track_id is not None:
                    print(f"🔒 Track ID {self.locked_track_id} confirmed! Stopping scan...")
                    return True
                    
                if self.state in ["CENTERING", "LOCKED"]:
                    print("🎯 Target confirmed!")
                    return True
                time.sleep(0.5)
        
        print("Search complete - no target found")
        return False
    
    def land(self):
        print("Landing...")
        self.vehicle.mode = VehicleMode("LAND")
        
        while self.vehicle.armed:
            alt = self.vehicle.location.global_relative_frame.alt
            print(f" Landing... Altitude: {alt:.2f}m")
            time.sleep(2)
    
    def close(self):
        self.vehicle.close()



class PersonDetector:
    """YOLOv5 detector with ByteTrack MOT integration"""
    
    def __init__(self, model_path='yolov5x.pt', conf_thres=0.20, imgsz=640):
        print("🔧 Initializing YOLOv5 detector with MOT...")
        
        # YOLOv5 setup
        self.device = select_device('')
        self.model = DetectMultiBackend(model_path, device=self.device, dnn=False)
        self.stride = self.model.stride
        self.names = self.model.names
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
        
        # YOLOv5 parameters - RELAXED for stability
        self.conf_thres = conf_thres  # 0.20 - detect more
        self.iou_thres = 0.40
        
        # Detection parameters
        self.allowed_classes = [0]  # Person class only
        self.min_detection_area = 1000  # Minimum bbox area
        self.min_confidence = 0.25  # Minimum confidence after NMS
        
        # ByteTrack MOT tracker - TUNED for drone movement scenarios
        self.tracker = ByteTracker(
            max_age=60,  # INCREASED: Keep track alive longer (60 frames = 3 seconds at 20fps)
            min_hits=3,   # INCREASED: Need 3 hits to confirm (more stable)
            iou_threshold=0.25  # LOWERED: More lenient matching (allows bigger bbox shifts from drone movement)
        )
        
        print(f"✅ Model loaded: {model_path}")
        print(f"   Detection: conf={self.conf_thres}, min_area={self.min_detection_area}")
        print(f"   Tracking: max_age=60 frames, min_hits=3, IoU=0.25")
        print("🎯 MOT ENABLED: Tracks will have persistent IDs across frames!")
        
        # Initialize Person Re-ID extractor for appearance-based matching
        self.reid_extractor = PersonReIDExtractor(device=self.device)
    
    def detect(self, frame):
        """
        Run YOLOv5 detection + ByteTrack tracking
        
        Returns:
            tracks: List of tracked objects [[x1, y1, x2, y2, track_id], ...]
            annotated_frame: Frame with visualization
        """
        # Prepare image
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if len(img.shape) == 3:
            img = img[None]  # Add batch dimension
        
        # YOLOv5 inference
        pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 
                                   classes=self.allowed_classes, max_det=10)
        
        # Process detections
        detections = []
        orig_h, orig_w = frame.shape[:2]
        
        for det in pred:
            if len(det):
                # Scale boxes back to original image size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Filter by area and confidence
                    if area >= self.min_detection_area and conf >= self.min_confidence:
                        detections.append([x1, y1, x2, y2, float(conf)])
        
        # Convert to numpy array
        if len(detections) > 0:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 5))
        
        # Update tracker with detections
        tracks = self.tracker.update(detections)
        
        # Annotate frame
        annotated_frame = frame.copy()
        
        # Get locked track ID from drone if available
        locked_id = None
        try:
            # We'll pass this from the callback
            if hasattr(self, '_locked_id_for_viz'):
                locked_id = self._locked_id_for_viz
        except:
            pass
        
        # Draw center cross and tolerance box
        img_h, img_w = frame.shape[:2]
        center_x, center_y = img_w // 2, img_h // 2
        
        cv2.drawMarker(annotated_frame, (center_x, center_y), (0, 255, 0), 
                      cv2.MARKER_CROSS, 40, 3)
        
        threshold_px = 40  # Match centering threshold
        cv2.rectangle(annotated_frame, 
                     (center_x - threshold_px, center_y - threshold_px),
                     (center_x + threshold_px, center_y + threshold_px),
                     (0, 255, 0), 2)
        
        # Draw all detections (light gray - raw YOLOv5)
        for det in detections:
            x1, y1, x2, y2, conf = det[:5]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
            cv2.putText(annotated_frame, f'Det:{conf:.2f}', (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Draw tracks (GREEN with persistent IDs)
        for track in tracks:
            x1, y1, x2, y2, track_id = track[:5]
            x1, y1, x2, y2, track_id = map(int, [x1, y1, x2, y2, track_id])
            
            # Check if this is the locked track
            is_locked = (locked_id is not None and track_id == locked_id)
            
            if is_locked:
                # Locked track: BRIGHT GREEN, thicker, with special label
                color = (0, 255, 0)
                thickness = 3
                label = f'🔒 LOCKED ID:{track_id}'
                label_color = (0, 255, 0)
                
                # Draw tracking line from center to target
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.line(annotated_frame, (center_x, center_y), (cx, cy), 
                        (0, 255, 255), 2)
                cv2.circle(annotated_frame, (cx, cy), 8, (0, 0, 255), -1)
            else:
                # Other tracks: normal green
                color = (0, 255, 0)
                thickness = 2
                label = f'ID:{track_id}'
                label_color = (0, 255, 0)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(annotated_frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
            
            # Draw center point
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)
        
        return tracks, annotated_frame


class DronePersonTracker:
    """Main system integrating detection, tracking, and drone control"""
    
    def __init__(self):
        print("=" * 60)
        print("🚁 DRONE PERSON TRACKER WITH MOT")
        print("=" * 60)
        
        # Initialize ROS
        rospy.init_node('drone_person_tracker_mot', anonymous=True)
        self.bridge = CvBridge()
        
        # Initialize components
        self.detector = PersonDetector(model_path='yolov5x.pt', conf_thres=0.20)
        self.drone = DroneController()
        
        # ROS subscriber
        self.image_sub = rospy.Subscriber("/webcam/image_raw", Image, 
                                         self.image_callback, queue_size=10)
        
        # Frame processing
        self.process_interval = 0.05  # 20 FPS max
        self.last_process_time = 0
        
        print("\n✅ System ready!")
        print("📹 Waiting for camera feed from /webcam/image_raw...")
        print("\n🎯 MOT TRACKING STRATEGY:")
        print("   1. Detect all people with YOLOv5")
        print("   2. Track them with persistent IDs (ByteTrack)")
        print("   3. Lock onto closest/largest person")
        print("   4. Follow that ID even through brief occlusions")
        print("=" * 60)
    
    def image_callback(self, data):
        """Process camera images with MOT tracking"""
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_process_time < self.process_interval:
            return
        self.last_process_time = current_time
        
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Pass locked ID to detector for visualization
            self.detector._locked_id_for_viz = self.drone.locked_track_id
            
            # Run detection + tracking
            tracks, annotated = self.detector.detect(cv_image)
            
            # Process tracks (pass cv_image for Re-ID feature extraction)
            self.process_tracks(tracks, cv_image)
            
            # Add state info to display
            state_text = f"State: {self.drone.state}"
            if self.drone.locked_track_id is not None:
                state_text += f" | Locked ID: {self.drone.locked_track_id}"
            cv2.putText(annotated, state_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Altitude and heading
            alt = self.drone.vehicle.location.global_relative_frame.alt
            heading = self.drone.vehicle.heading
            info_text = f"Alt: {alt:.1f}m | Heading: {heading:.0f}deg"
            cv2.putText(annotated, info_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Distance if available
            if self.drone.estimated_distance:
                dist_text = f"Distance: {self.drone.estimated_distance:.1f}m"
                cv2.putText(annotated, dist_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
            
            # Display
            cv2.imshow("MOT Tracking", annotated)
            cv2.waitKey(1)
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
    
    def process_tracks(self, tracks, cv_image):
        """Process tracked objects and control drone"""
        height, width = cv_image.shape[:2]
        
        if len(tracks) == 0:
            # No tracks
            self.drone.track_lost_count += 1
            
            # Only warn if we had a locked track
            if self.drone.locked_track_id is not None:
                if self.drone.track_lost_count > self.drone.max_track_lost_frames:
                    rospy.logwarn(f"❌ Lost track ID {self.drone.locked_track_id} for too long!")
                    self.drone.locked_track_id = None
                    self.drone.state = "SEARCHING"
            return
        
        # Reset lost count
        self.drone.track_lost_count = 0
        
        # If we have a locked track, find it
        if self.drone.locked_track_id is not None:
            locked_track = None
            for track in tracks:
                if int(track[4]) == self.drone.locked_track_id:
                    locked_track = track
                    break
            
            if locked_track is not None:
                # Found our locked track - follow it!
                x1, y1, x2, y2 = map(int, locked_track[:4])
                self.follow_track(x1, y1, x2, y2, width, height, cv_image)
                # Only log occasionally to avoid spam
                if int(time.time() * 4) % 4 == 0:  # Every 0.25s
                    rospy.loginfo(f"🎯 Following locked track ID {self.drone.locked_track_id}")
            else:
                # Locked track not visible this frame (occlusion?)
                # Only warn occasionally
                if self.drone.track_lost_count % 20 == 0:  # Every 20 frames (~1 second)
                    rospy.logwarn(f"⚠️ Track ID {self.drone.locked_track_id} occluded for {self.drone.track_lost_count} frames...")
        else:
            # No lock yet - only lock if we're in SEARCHING state
            # This prevents locking during rotation
            if self.drone.state == "SEARCHING":
                # Select largest track (closest person)
                largest_track = max(tracks, key=lambda t: (t[2]-t[0])*(t[3]-t[1]))
                x1, y1, x2, y2, track_id = map(int, largest_track[:5])
                
                # Get the actual tracker object to check hit count
                track_confirmed = False
                for trk in self.detector.tracker.trackers:
                    if trk.id == track_id:
                        # Only lock if track has been seen multiple times (min_hits=3)
                        if trk.hit_streak >= 3:  # Match min_hits parameter
                            track_confirmed = True
                            rospy.loginfo(f"✓ Track ID {track_id} confirmed (hit_streak={trk.hit_streak})")
                        else:
                            rospy.loginfo(f"Track ID {track_id} hit_streak={trk.hit_streak}, need >= 3")
                        break
                
                if track_confirmed:
                    # Lock onto this track
                    self.drone.locked_track_id = track_id
                    rospy.logwarn(f"🔒 LOCKED onto track ID {track_id}!")
                    self.drone.state = "CENTERING"
                    
                    # Reset altitude tracking for new lock
                    self.drone.altitude_at_lock = self.drone.vehicle.location.global_relative_frame.alt
                    self.drone.cumulative_altitude_change = 0.0
                    rospy.loginfo(f"📍 Altitude at lock: {self.drone.altitude_at_lock:.2f}m")
                    
                    # ===== EXTRACT TARGET APPEARANCE FEATURES FOR RE-ID =====
                    if self.detector.reid_extractor is not None:
                        bbox = [x1, y1, x2, y2]
                        self.drone.target_features = self.detector.reid_extractor.extract_features(cv_image, bbox)
                        rospy.loginfo(f"🎨 Extracted target appearance features (dim={len(self.drone.target_features)})")
                        rospy.loginfo(f"   Feature norm: {np.linalg.norm(self.drone.target_features):.4f}")
                    
                    self.follow_track(x1, y1, x2, y2, width, height, cv_image)
                else:
                    # Track not confirmed yet, keep searching
                    pass  # Already logged above
    
    def follow_track(self, x1, y1, x2, y2, width, height, cv_image):
        """Follow a specific tracked person with full PID control + Re-ID verification"""
        # Update last detection time
        self.drone.last_detection_time = time.time()
        
        # ===== PERSON RE-ID VERIFICATION =====
        # ONLY verify during blind approach phase (< 3m) when we're not centering
        # During centering, person appearance changes with drone angle - unreliable!
        if (self.drone.in_blind_approach and 
            self.drone.target_features is not None and 
            self.detector.reid_extractor is not None):
            
            bbox = [x1, y1, x2, y2]
            current_features = self.detector.reid_extractor.extract_features(cv_image, bbox)
            similarity = PersonReIDExtractor.compute_similarity(self.drone.target_features, current_features)
            
            # Log Re-ID match quality (occasionally)
            if int(time.time() * 2) % 10 == 0:  # Every 5 seconds
                if similarity >= self.drone.reid_threshold:
                    rospy.loginfo(f"✅ Re-ID Match: {similarity:.2f} (threshold={self.drone.reid_threshold:.2f})")
                else:
                    rospy.logwarn(f"⚠️ Re-ID Low: {similarity:.2f} < {self.drone.reid_threshold:.2f} - Possible wrong person!")
            
            # If similarity is too low, this might not be our target
            if similarity < self.drone.reid_threshold * 0.6:  # 60% of threshold (more lenient)
                rospy.logwarn(f"❌ Re-ID Failed! Similarity={similarity:.2f} - Unlocking track")
                self.drone.locked_track_id = None
                self.drone.target_features = None
                self.drone.state = "SEARCHING"
                return
        
        # Calculate center offset
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        x_offset = cx - width // 2
        y_offset = cy - height // 2
        
        # Calculate bbox dimensions for edge detection
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        distance = self.drone.estimate_distance(bbox_height, height)
        
        # State transitions
        if self.drone.state == "SEARCHING":
            rospy.logwarn(f"⚠️⚠️ TRACK ID {self.drone.locked_track_id} DETECTED! ⚠️⚠️")
            self.drone.state = "DETECTED"
            time.sleep(0.5)
        
        if self.drone.state == "DETECTED":
            rospy.logwarn("🎯 Starting CENTERING...")
            self.drone.state = "CENTERING"
        
        # Display info
        rospy.loginfo(f"Track ID {self.drone.locked_track_id}: X={x_offset:+4d}px Y={y_offset:+4d}px | "
                     f"Dist={distance:.1f}m | State={self.drone.state}")
        
        # Center the target with PID control (only if in CENTERING state)
        if self.drone.state == "CENTERING":
            is_locked = self.drone.center_target(x_offset, y_offset, width, height, bbox_height, bbox_width)
            
            if is_locked:
                # CENTERING COMPLETE - Transition to TRAJECTORY mode
                rospy.logwarn("="*70)
                rospy.logwarn(f"✓✓✓ TRACK ID {self.drone.locked_track_id} CENTERED! ✓✓✓")
                rospy.logwarn("🎯 Entering HYBRID TRAJECTORY MODE...")
                rospy.logwarn("="*70)
                
                # Calculate precise trajectory
                self.drone.trajectory_forward_m, self.drone.trajectory_lateral_m, self.drone.trajectory_vertical_m = \
                    self.drone.calculate_precise_trajectory(x_offset, y_offset, width, height, bbox_height)
                
                if self.drone.trajectory_forward_m is not None:
                    self.drone.trajectory_calculated = True
                    self.drone.state = "TRAJECTORY_EXEC"
                else:
                    rospy.logwarn("⚠️ Trajectory calculation failed - falling back to visual servoing")
                    self.drone.state = "APPROACHING"
        
        # Execute trajectory with hybrid edge monitoring
        elif self.drone.state == "TRAJECTORY_EXEC":
            # Create bbox tracker function for edge monitoring
            current_bbox_data = [x1, y1, x2, y2]
            bbox_tracker = lambda: current_bbox_data
            
            rospy.loginfo("🚀 Executing hybrid trajectory (smooth movement + edge safety)...")
            
            # Execute trajectory with edge monitoring
            success = self.drone.execute_precise_trajectory(
                bbox_tracker=bbox_tracker,
                image_width=width,
                image_height=height
            )
            
            if success:
                rospy.logwarn("✓✓✓ TRAJECTORY COMPLETE! ✓✓✓")
                self.drone.approach_complete = True
                self.drone.state = "LOCKED"
            else:
                rospy.logwarn("⚠️ Trajectory execution interrupted")
        
        # If in APPROACHING state, continue the approach process
        elif self.drone.state == "APPROACHING":
            # Let center_target() handle the approach logic (fallback for visual servoing)
            is_locked = self.drone.center_target(x_offset, y_offset, width, height, bbox_height, bbox_width)
            
            if is_locked:
                self.drone.state = "LOCKED"
                rospy.loginfo(f"✓✓✓ APPROACH COMPLETE - TRACK ID {self.drone.locked_track_id} LOCKED! ✓✓✓")


def main():
    print("\n" + "="*70)
    print("🎯 MOT-BASED DRONE PERSON TRACKER")
    print("YOLOv5 + ByteTrack Multi-Object Tracking")
    print("Persistent ID tracking - No more detection flicker!")
    print("="*70 + "\n")
    
    try:
        # Initialize
        tracker = DronePersonTracker()
        
        print(f"\n🔧 Configuration:")
        print(f"   Centering Threshold: ±{tracker.drone.centering_threshold_pixels}px")
        print(f"   Approach Tolerance: ±{tracker.drone.approach_tolerance}px")
        print(f"   YAW PID: Kp={tracker.drone.pid_yaw_kp} Ki={tracker.drone.pid_yaw_ki} Kd={tracker.drone.pid_yaw_kd}")
        print(f"   ALT PID: Kp={tracker.drone.pid_alt_kp} Ki={tracker.drone.pid_alt_ki} Kd={tracker.drone.pid_alt_kd}")
        print(f"   Distance: Target={tracker.drone.target_approach_distance}m (70cm)")
        print(f"   Hold Time: {tracker.drone.centered_hold_time}s")
        print(f"   MOT: max_age=60 frames, min_hits=3, IoU=0.25 (DRONE-TUNED)\n")
        
        # Takeoff
        tracker.drone.arm_and_takeoff(10)
        
        # Start ROS processing
        ros_thread = threading.Thread(target=rospy.spin)
        ros_thread.daemon = True
        ros_thread.start()
        
        print("\n--- Starting Mission ---\n")
        
        # Search for target
        tracker.drone.state = "SEARCHING"
        found = tracker.drone.rotate_and_search(rotation_steps=24, pause_time=3)
        
        if found:
            print("\n🎯 TARGET FOUND! Tracking with MOT...")
            
            # Wait for approach completion or timeout
            timeout = 120
            start = time.time()
            
            while time.time() - start < timeout:
                # Check if approach was completed successfully
                if tracker.drone.approach_complete and not tracker.drone.hovering_complete:
                    # HOVERING PHASE: Stay at target for 10 seconds using ZERO VELOCITY (not ALT_HOLD)
                    # USER REQUIREMENT: "don't put it to hover mode (alt_hold) as it cause the drop to fall down"
                    if tracker.drone.hover_start_time is not None:
                        hover_elapsed = time.time() - tracker.drone.hover_start_time
                        print(f"\n⏱️ HOVERING at target (5m)... ({hover_elapsed:.1f}/{tracker.drone.hover_duration:.0f}s)")
                        
                        # Send ZERO VELOCITY commands to hold position (stay in GUIDED mode)
                        # This prevents the drone from falling like ALT_HOLD does
                        msg = tracker.drone.vehicle.message_factory.set_position_target_local_ned_encode(
                            0, 0, 0,
                            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                            0b0000111111000111,  # Use velocity
                            0, 0, 0,  # Position (not used)
                            0, 0, 0,  # ZERO velocity - hold position!
                            0, 0, 0,  # Acceleration (not used)
                            0, 0)     # Yaw (not used)
                        tracker.drone.vehicle.send_mavlink(msg)
                        tracker.drone.vehicle.flush()
                        
                        if hover_elapsed >= tracker.drone.hover_duration:
                            print(f"\n✓✓✓ HOVER COMPLETE ({tracker.drone.hover_duration:.0f}s)! ✓✓✓")
                            tracker.drone.hovering_complete = True
                
                elif tracker.drone.hovering_complete and not tracker.drone.rtl_complete:
                    # RTL PHASE: Return to launch position
                    print(f"\n🏠 Initiating Return-To-Launch...")
                    tracker.drone.return_to_launch()
                    print(f"\n✓✓✓ RTL COMPLETE! ✓✓✓")
                    break
                
                elif tracker.drone.approach_complete and tracker.drone.hovering_complete and tracker.drone.rtl_complete:
                    # All phases complete
                    print(f"\n✓✓✓ MISSION COMPLETE! ✓✓✓")
                    break
                # Check if track was lost
                elif tracker.drone.state == "SEARCHING":
                    elapsed = time.time() - start
                    print(f"\n⚠️ Track lost after {elapsed:.1f}s - Attempting to reacquire...")
                    # Give it 10 more seconds to reacquire
                    if elapsed > 30:  # If lost for more than 30s total, give up
                        print("⚠️ Could not reacquire target - mission incomplete")
                        break
                time.sleep(1)
        else:
            print("\n⚠️ No target found in initial search")
        
        print("\n--- Mission Complete ---")
        
        # ===== INTELLIGENT LANDING DECISION =====
        # Mission sequence: Approach (0.7m) → Hover (10s) → RTL → Land
        
        should_land = False
        
        if tracker.drone.rtl_complete:
            # Full mission success: approach + hover + RTL complete
            print(f"\n✅✅✅ MISSION SUCCESS ✅✅✅")
            print(f"   ✓ Approached to {tracker.drone.target_approach_distance}m")
            print(f"   ✓ Hovered for {tracker.drone.hover_duration:.0f} seconds")
            print(f"   ✓ Returned to launch position")
            print(f"   Track ID: {tracker.drone.locked_track_id}")
            print(f"   Final approach distance: {tracker.drone.estimated_distance:.1f}m")
            should_land = True
            time.sleep(2)
        elif tracker.drone.approach_complete and tracker.drone.hovering_complete:
            # Approach and hover complete, but RTL failed
            print(f"\n⚠️ PARTIAL SUCCESS - RTL incomplete")
            print(f"   ✓ Reached target distance and hovered")
            print(f"   ⚠️ Return-to-launch incomplete")
            print(f"   Landing at current position for safety...")
            should_land = True
        elif tracker.drone.approach_complete:
            # Approach complete but hover incomplete
            print(f"\n⚠️ PARTIAL SUCCESS - Hover incomplete")
            print(f"   ✓ Reached target distance")
            print(f"   ⚠️ Hover interrupted")
            print(f"   Landing at current position for safety...")
            should_land = True
        elif not found:
            print("\n⚠️ Mission INCOMPLETE - No target found")
            print("   Landing for safety...")
            should_land = True
        else:
            print("\n⚠️ Mission INCOMPLETE - Track lost before reaching target")
            print(f"   Current distance: {tracker.drone.estimated_distance:.1f}m (target: {tracker.drone.target_approach_distance}m)")
            print("   🚁 HOLDING POSITION - Not landing!")
            print("   Drone will hover and maintain altitude")
            print("   Press Ctrl+C to manually land when ready")
            should_land = False
            
            # Hold position and wait for user intervention
            print("\n⏸️ Entering HOVER mode...")
            tracker.drone.vehicle.mode = VehicleMode("ALT_HOLD")
            
            # Keep spinning ROS to maintain connection
            while True:
                time.sleep(1)
                if rospy.is_shutdown():
                    break
        
        # Only land if mission complete or safety required
        if should_land:
            tracker.drone.land()
            
            print("\n" + "="*70)
            print("LANDING COMPLETE")
            print("="*70 + "\n")
        
        tracker.drone.close()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
