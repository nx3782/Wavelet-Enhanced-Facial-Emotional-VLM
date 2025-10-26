import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import pywt
from pathlib import Path
from tqdm import tqdm

def extract_wavelet_features(trajectory, n_coefficients=10):
    """
    Convert landmark trajectory to fixed-length wavelet features.
    
    Args:
        trajectory: (T, 3) array - T frames, 3 coordinates (x,y,z)
        n_coefficients: number of output coefficients
    
    Returns:
        (n_coefficients, 3) array
    """
    features = []
    
    for coord_idx in range(3):  # x, y, z
        signal = trajectory[:, coord_idx]
        
        # Determine decomposition level
        T = len(signal)
        T_prime = 2 ** int(np.ceil(np.log2(T)))
        L = max(1, int(np.log2(T_prime / n_coefficients)))
        
        # Wavelet decomposition
        coeffs = pywt.wavedec(signal, 'haar', level=L)
        approximation = coeffs[0]
        
        # Normalize
        normalized = approximation / np.sqrt(2 ** L)
        
        # Pad or truncate to n_coefficients
        if len(normalized) < n_coefficients:
            normalized = np.pad(normalized, (0, n_coefficients - len(normalized)))
        else:
            normalized = normalized[:n_coefficients]
        
        features.append(normalized)
    
    # Stack: (n_coefficients, 3)
    return np.stack(features, axis=1)


def process_video_to_wavelet(video_path, n_coefficients=10):
    """
    Extract landmarks from video and convert to wavelet features.
    
    Returns:
        (n_coefficients, 478, 3) array or None if failed
    """
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    landmarks_per_frame = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Extract 478 landmarks
            face = results.multi_face_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face.landmark])
            landmarks_per_frame.append(landmarks)
    
    cap.release()
    face_mesh.close()
    
    if len(landmarks_per_frame) < 10:
        return None
    
    # Shape: (T_frames, 478, 3)
    landmarks_array = np.array(landmarks_per_frame)
    
    # Apply wavelet transform to each landmark
    wavelet_features = []
    for landmark_idx in range(478):
        trajectory = landmarks_array[:, landmark_idx, :]  # (T_frames, 3)
        wavelet_feat = extract_wavelet_features(trajectory, n_coefficients)
        wavelet_features.append(wavelet_feat)
    
    # Stack: (478, n_coefficients, 3) -> transpose to (n_coefficients, 478, 3)
    result = np.stack(wavelet_features, axis=1)
    
    return result


def create_landmark_dataset(csv_path, output_path, n_coefficients=10):
    """
    Process all videos and create the landmark NPY file.
    
    Args:
        csv_path: CSV with columns ['video_idx', 'file_path', ...]
        output_path: Where to save the output NPY file
        n_coefficients: Number of wavelet coefficients (default: 10)
    """
    df = pd.read_csv(csv_path)
    
    # Sort by video_idx to ensure order
    df = df.sort_values('video_idx').reset_index(drop=True)
    
    all_landmarks = []
    failed_videos = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_path = row['file_path']
        
        wavelet_features = process_video_to_wavelet(video_path, n_coefficients)
        
        if wavelet_features is None:
            print(f"Failed: {video_path}")
            failed_videos.append(video_path)
            # Use zeros as placeholder
            wavelet_features = np.zeros((n_coefficients, 478, 3))
        
        all_landmarks.append(wavelet_features)
    
    # Stack all videos: (N, 10, 478, 3)
    all_landmarks = np.array(all_landmarks)
    
    # Save
    np.save(output_path, all_landmarks)
    
    print(f"\n{'='*60}")
    print(f"Saved landmark data: {output_path}")
    print(f"Shape: {all_landmarks.shape}")
    print(f"Failed videos: {len(failed_videos)}/{len(df)}")
    print(f"{'='*60}")
    
    if failed_videos:
        print("\nFailed videos:")
        for v in failed_videos[:10]:
            print(f"  - {v}")


if __name__ == "__main__":
    # Usage
    create_landmark_dataset(
        csv_path='data/train.csv',
        output_path='data/facial_landmarks_wavelet.npy',
        n_coefficients=10
    )
