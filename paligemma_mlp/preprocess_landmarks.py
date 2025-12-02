import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pywt
from pathlib import Path
from tqdm import tqdm


def extract_wavelet_features(trajectory, n_coefficients=10):
    """
    Convert trajectory to fixed-length wavelet features.
    
    Args:
        trajectory: (T, D) array where T=frames, D=dimensions
        n_coefficients: number of output coefficients
    
    Returns:
        (n_coefficients, D) array
    """
    features = []
    num_dims = trajectory.shape[1]
    
    for dim_idx in range(num_dims):
        signal = trajectory[:, dim_idx]
        
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
    
    return np.stack(features, axis=1)


class FaceLandmarkerWithBlendshapes:
    """Wrapper for MediaPipe FaceLandmarker that supports blendshapes"""
    
    def __init__(self, model_path='face_landmarker.task'):
        """Initialize FaceLandmarker with blendshapes support in VIDEO mode."""
        if not Path(model_path).exists():
            print(f"\n{'='*60}")
            print(f"WARNING: Model file not found: {model_path}")
            print(f"Blendshapes will NOT be extracted!")
            print(f"{'='*60}\n")
            self.detector = None
            return
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # ← FIX: VIDEO mode
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    def process(self, frame, timestamp_ms):
        """Process a frame and return landmarks + blendshapes"""
        if self.detector is None:
            return None
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        return detection_result
    
    def close(self):
        if self.detector:
            self.detector.close()

# use np_filed[index]["file_path"] to check whether it exists. 
def process_video(video_path, n_coefficients=10, model_path='face_landmarker.task'):
    """
    Extract landmarks and blendshapes from video, convert to wavelet features.
    
    Returns:
        dict with 'file_path', 'landmarks', 'blendshapes' or None if failed
    """
    use_new_api = Path(model_path).exists()
    
    if use_new_api:
        detector = FaceLandmarkerWithBlendshapes(model_path)
        if detector.detector is None:
            use_new_api = False
    
    if not use_new_api:
        print(f"Using FaceMesh (no blendshapes) for: {video_path}")
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if use_new_api:
            detector.close()
        else:
            face_mesh.close()
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    
    landmarks_per_frame = []
    blendshapes_per_frame = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if use_new_api:
            timestamp_ms = int((frame_idx / fps) * 1000)
            results = detector.process(rgb_frame, timestamp_ms)
            
            if results and results.face_landmarks:
                face = results.face_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face])
                landmarks_per_frame.append(landmarks)
                
                if results.face_blendshapes:
                    blendshapes = np.array([bs.score for bs in results.face_blendshapes[0]])
                    blendshapes_per_frame.append(blendshapes)
                else:
                    blendshapes_per_frame.append(np.zeros(52))
        else:
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face.landmark])
                landmarks_per_frame.append(landmarks)
                blendshapes_per_frame.append(np.zeros(52))
        
        frame_idx += 1
    
    cap.release()
    if use_new_api:
        detector.close()
    else:
        face_mesh.close()
    
    if len(landmarks_per_frame) < 10:
        return None
    
    landmarks_array = np.array(landmarks_per_frame)
    blendshapes_array = np.array(blendshapes_per_frame)
    
    # Apply wavelet transform to each landmark
    wavelet_landmarks = []
    for landmark_idx in range(478):
        trajectory = landmarks_array[:, landmark_idx, :]
        wavelet_feat = extract_wavelet_features(trajectory, n_coefficients)
        wavelet_landmarks.append(wavelet_feat)
    
    landmark_features = np.stack(wavelet_landmarks, axis=1)
    blendshape_features = extract_wavelet_features(blendshapes_array, n_coefficients)
    
    return {
        'file_path': str(video_path),
        'landmarks': landmark_features,
        'blendshapes': blendshape_features
    }


def create_combined_dataset(csv_path, output_path, n_coefficients=10, model_path='face_landmarker.task'):
    """
    Process all videos and save to a single NPY file.
    
    Args:
        csv_path: CSV with 'file_path' column
        output_path: Path to save the combined .npy file
        n_coefficients: Number of wavelet coefficients (default: 10)
        model_path: Path to MediaPipe FaceLandmarker model (for blendshapes)
    """
    df = pd.read_csv(csv_path)
    
    if 'file_path' not in df.columns:
        raise ValueError("CSV must contain 'file_path' column")
    
    if 'video_idx' in df.columns:
        df = df.sort_values('video_idx').reset_index(drop=True)
    
    all_data = []
    failed_videos = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_path = row['file_path']
        
        features = process_video(video_path, n_coefficients, model_path)
        
        if features is None:
            print(f"\nFailed: {video_path}")
            failed_videos.append(video_path)
            features = {
                'file_path': str(video_path),
                'landmarks': np.zeros((n_coefficients, 478, 3)),
                'blendshapes': np.zeros((n_coefficients, 52))
            }
        
        all_data.append(features)
    
    all_data = np.array(all_data, dtype=object)
    np.save(output_path, all_data)
    
    print(f"\n{'='*60}")
    print(f"Saved combined dataset: {output_path}")
    print(f"Total videos: {len(all_data)}")
    print(f"Successful: {len(all_data) - len(failed_videos)}/{len(df)}")
    print(f"Failed: {len(failed_videos)}/{len(df)}")
    print(f"{'='*60}")
    
    if failed_videos:
        print("\nFailed videos:")
        for v in failed_videos[:10]:
            print(f"  - {v}")
        if len(failed_videos) > 10:
            print(f"  ... and {len(failed_videos) - 10} more")


def load_and_verify(npy_path, video_idx=0):
    """Load and verify the structure of the combined NPY file."""
    data = np.load(npy_path, allow_pickle=True)
    
    print(f"\n{'='*60}")
    print(f"File: {npy_path}")
    print(f"Total videos: {len(data)}")
    print(f"\nInspecting video at index {video_idx}:")
    
    video_data = data[video_idx]
    
    print(f"File path: {video_data['file_path']}")
    print(f"Landmarks shape: {video_data['landmarks'].shape}")
    print(f"Blendshapes shape: {video_data['blendshapes'].shape}")
    
    # Check if blendshapes are actually extracted
    if np.all(video_data['blendshapes'] == 0):
        print("\n⚠️  WARNING: Blendshapes are all zeros!")
        print("This means blendshapes were not extracted.")
    else:
        print("\n✓ Blendshapes successfully extracted!")
        print(f"Non-zero blendshape values: {np.count_nonzero(video_data['blendshapes'])}")
    
    print(f"\nSample landmarks (first coefficient, first 3 landmarks):")
    print(video_data['landmarks'][0, :3, :])
    
    print(f"\nSample blendshapes (first coefficient, first 10 values):")
    print(video_data['blendshapes'][0, :10])
    print(f"{'='*60}\n")

# example usage
if __name__ == "__main__":
    model_path = 'face_landmarker.task'
    
    if not Path(model_path).exists():
        print("Downloading FaceLandmarker model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"Downloaded to: {model_path}")
    
    # Process videos from csv file
    create_combined_dataset(
        csv_path='../data/DFEW/sample_DFEW_emo_label.csv',
        output_path='../data/DFEW/sample_facial_landmarks_wavelet.npy',
        n_coefficients=10,
        model_path=model_path
    )
    
    # Verify
    load_and_verify('../data/DFEW/sample_facial_landmarks_wavelet.npy', video_idx=0)
