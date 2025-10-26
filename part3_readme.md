Part 3: First Update

**Methods**:
1.1 Data Input: In this section, we obatin video data from the DFEW dataset. As described in earlier part, these input videos contain clear facial expressions and movements along with true emotion labels. We may only use this dataset for training purpose because this contains rich amount of data, around 11k and also clear 7 types of emotion label for all videos. 
Metadata: We create a CSV file containing video IDs, video paths, and true emotional labels to keep the data clean and easy to use in the future. We have a total of 11697 video inputs with an average length of 10 seconds for use. There are total 9356 training data and 2341 test data, which follows a 80-20 train/test split here. We have attached the sample csv file for better understanding on the data structure. 
Pre-extracted Features: We have extracted facial landmarks (478 3D points) and blendshapes (52 facial expression coefficients) extracted using MediaPipe for each video and store them inside the npy file. Example npy file is also given in this repo for quick view. 

1.2 Facial Feature Extraction: For this part, we use MediaPipe's FaceLandmarker to extract two types of features from each video frame:
Landmarks: (478 points × 3 coordinates), 3D coordinates (x, y, z) for 478 facial points, which captures geometric structure of the face and tracks spatial positions of key facial features (eyes, mouth, nose,etc.). Each of the 478 landmarks is tracked across all video frames, creating a temporal trajectory for each coordinate (x, y, z). We apply Haar wavelet decomposition separately to each coordinate's trajectory, compressing it from T frames to 10 coefficients (we can adjust this). This results in a final shape of (10, 478, 3) where each landmark's temporal movement is represented by 10 wavelet coefficients per coordinate.

Blendshapes: (52 coefficients), this is a semantic representation of facial expressions. Each coefficient represents a specific facial action (smile, frown, eyebrow raise, etc.) A pre-normalized expression space that generalizes across individuals. Each of the 52 blendshape values is tracked across all video frames, creating 52 temporal signals. We apply Haar wavelet decomposition to each blendshape's trajectory independently, compressing each from T frames to 10 coefficients. This results in a final shape of (10, 52) where each blendshape's temporal activation pattern is represented by 10 wavelet coefficients.

In both cases, the wavelet transform solves the variable-length problem: whether a video has 300 frames or 1800 frames, we always get exactly 10 coefficients, making it possible to feed fixed-size features into the neural network while preserving temporal dynamics at multiple time scales.

**Justification**:
2.1 Why MediaPipe for Feature Extraction: This is an industry-standard tool for facial landmark detection. It provides both geometric (landmarks) and semantic (blendshapes) representations with strong real-time performance for large-scale processing. It is robust to lighting conditions and head poses. We have more landmarks (478) compared to traditional methods (68 points). MediaPipe was pre-trained on diverse datasets ensuring generalization. Blendshapes also provide interpretable, normalized expression space.

2.2 Why Wavelet Transform for Temporal Encoding
Problem: Videos have variable durations (ranging from a few seconds to 60+ seconds), but neural networks require fixed-size inputs.
Thus, we want to try using wavelets to solve this. Why it work here? This gives us fixed-length output: Regardless of video duration, we get exactly 10 coefficients or more based on the setup. It also gives multi-scale information preservation: Captures both slow trends (approximation) and rapid changes (details). This is computationally efficient with O(n) complexity, much faster than recurrent networks. Also it does better at information preservation compared to traiditional averaging or uniform sampling, which guarantees lots of information loss on temporal dynamics, decent amount of frames even though they can be redundant.  

2.3 Why Combine Landmarks and Blendshapes
Complementary information: While landmarks capture individual-specific facial geometry and person-dependent patterns which help prevent overfit on certain facial structures only, blendshapes capture expression semantics in a normalized, person-independent space, which help prevent missing subtle geometric variations on the face. By combining both extra features here, they provide both "what the face looks like" and "what expression is being made" with explicit distance information also available. 

2.4 Why Multi-Modal Fusion:
Visual content provides context: Input videos give us background information, lighting, environment, and overall scene understanding. We also have non-facial cues (body language, posture).

Landmarks provide precision: We have exact facial movements and micro-expressions, We also have temporal dynamics of expressions. Additionally we have fine-grained emotion indicators.
Cross-attention benefits: This allows landmark features to attend to relevant visual regions. It also enables text to guide which facial features to focus on. Also it creates synergistic representations which are stronger than concatenation.

2.5 Why Do We use PaliGemma-based Architecture
Foundation model advantages: PaliGemma is pre-trained on large-scale vision-language data, which means it has ttrong visual understanding and reasoning capabilities. Also this is efficient in inference with 3B parameters. 

Prefix tuning + LoRA benefits: Parameter efficiency: Only ~118M trainable parameters vs. 3.1B total. We maintain pre-trained knowledge while adapting to new domain. This gives faster training and lower memory requirements while preventing the model from catastrophically forgetting.


**Demo**: Core columns inside the cleaned CSV file:
- file_path: actual path to video file
- video_idx: unique identifier for indexing
- label: numerical label indicating the emotion type
- actual: actual text label indicating the emotion of the person in the video

We also expect to have prompt included, but we are still testing out the performance of this input as we believe the facial emotional changes should be the primary focus for this project. We decide to exclude that for now. 


**Output Format**: NPY file containing array of dictionaries, example views are also given here for better structure view:
Each entry: {
    'file_path': str,
    'landmarks': (10, 478, 3) array,
    'blendshapes': (10, 52) array
}

### 3.2 Wavelet Feature Extraction

**Configuration:**
- Wavelet type: Haar (simplest, computationally efficient)
- Number of coefficients: 10 (can adjust)
- Normalization: Coefficients divided by √(2^L)

**Adaptive decomposition level**:
```
For video with T frames:
T' = 2^⌈log₂(T)⌉
L = ⌊log₂(T' / 10)⌋

Example:
- 300 frames → T'=512 → L=5 → ~10 coefficients
- 1800 frames → T'=2048 → L=7 → ~8 coefficients (padded to 10)


### 3.3 Model Training Configuration for now (still under tuning)

**Model:** PaliGemma-2-3B (mix-448)
- Vision encoder: SigLIP
- Language model: Gemma-2-2B
- Input resolution: 448×448

**Training Strategy:**
- Optimizer: AdamW
- Learning rate: 5e-5 with warmup
- Batch size: 2-4 (limited by GPU memory)
- Training scheme: Regression on PROMIS scores

**Efficient Fine-tuning:**
- LoRA rank: 8-16
- LoRA alpha: 16-32
- Target modules: Query and Value projections
- Trainable parameters: ~118M / 3.1B (~3.8%)

### 3.4 Data Quality Considerations

**Face detection quality:**
- Minimum frames required: 10 frames with detected faces
- Quality threshold: At least 90% of frames should have detectable faces
- Failed videos: Filled with zero features and flagged for review

**Temporal consistency:**
- MediaPipe's VIDEO mode ensures temporal smoothness
- Tracking across frames reduces jitter
- Timestamp-based processing maintains temporal ordering

**Handling edge cases:**
- Occluded faces: MediaPipe interpolates missing landmarks
- Multiple faces: Only process primary face (closest to center)
- Poor lighting: MediaPipe's robust detection handles moderate lighting variations
- 

## 4. Expected Performance Characteristics

### 4.1 Feature Dimensionality

**Compression achieved:**
Before wavelet transform, we have:
- 60-second video at 30 FPS = 1800 frames
- 478 landmarks × 3 coords × 1800 frames = 2,581,200 values

After wavelet transform, we have:
- 478 landmarks × 3 coords × 10 coeffs = 14,340 values
- Compression ratio: 180x smaller

Blendshapes:
- 52 blendshapes × 1800 frames = 93,600 values
- After wavelets: 52 × 10 = 520 values
- Compression ratio: 180x smaller
```

**Contribution**: This is an individual final project work. So I have done data curation and preparation, including data cleaning and data pre-processing alone throughout the semester. 

**Instructions on Codes**: First git clone this repo and run "cd Wavelet-Enhanced-Facial-Emotional-VLM" inside Xcode or other environments. 

For the data pre-process part, stay in the main directory, and run "python data_clean.py" or "python3 data_clean.py". File paths are already set up for easy check. 
This scripts does the following task:
1. Read in the raw csv file that contains columns 1happy, 2sad, 3neutral, 4angry, 5surprise, 6disgust, 7fear, order, label, file_path.
2. Convert the numerical label from column "label" into the actual text label using the numerical indicator in the front of each label in other columns.
3. Add a new column named "video_id" for better view. This is the same as "order" but to make it clean. 
4. After mapping is done, save all mapped text labels into a new column named "actual".
5. Save the csv file under pre-defined output directory.

Here is the original csv structure: 

<tr><td width="30%"><image src="samples-gif/original_csv.png" /></td><td width="15%"></td></tr>


You can expect the following output structure in the csv file after this mapping.

<tr><td width="30%"><image src="samples-gif/label_map.png" /></td><td width="15%"></td></tr>


For the landmarks and blendshapes generation part, change directory into "paligema-mlp" folder, run "preprocess_landmarks.py" file to get landmarks and blendshapes packed into a npy file. The file path has already setup, so no need to change them manually as long as the file is running inside "paligema-mlp" folder. More specifically, the python file follows the following steps to get the landmark and blendshape data from input videos: 
1. Read in the pre-process csv file that contains columns 1happy, 2sad, 3neutral, 4angry, 5surprise, 6disgust, 7fear, order, label, actual, video_id, file_path. 
2. Load video using OpenCV
3. Extract frames and convert to RGB
4. Process each frame through MediaPipe FaceLandmarker
5. Collect landmarks (478, 3) and blendshapes (52,) per frame
6. Apply wavelet transform to temporal trajectories
7. Save compressed features to NPY file with pre-defined output path. 

After finishing running the code above, you can use np.load(file_name, allow_pickle=True) to see some outputs with similar structure to the following example view: 

<tr><td width="30%"><image src="samples-gif/npy_example_1.png" /></td><td width="15%"></td></tr>
<tr><td width="30%"><image src="samples-gif/npy_example_2.png" /></td><td width="15%"></td></tr>
<tr><td width="30%"><image src="samples-gif/npy_example_3.png" /></td><td width="15%"></td></tr>
