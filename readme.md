Part 1: Conceptual_design

**Abstract**: With the fast development of Large Language Models (LLMs), people have lots of mature and powerful AI tools to help them not only solve daily matters but also problems that are really challenging; for example, a series of mathematical problems that the latter problems one depend on the answers of the previous ones and step by step mathematical proofs that require strong logics. With such powerful LLMs as great backbones, computer scientists start to shift their focus more on Vision-Language models (VLMs) and also Multi-modal Models (MMs) to help machines get more useful information to better assist humans in all kinds of situations.

**Problems**: Extracting features from vision inputs has been one of the greatest challenges for these models, especially long video sequence inputs due to the fact that most state-of-the-art (SOTA) models still random sample frames from each second of the video input to reduce memory burden, which leads to unavoidable information loss, and simply selecting more frames will exponentially increase the training and inference time. 

**Solution**: To solve this problem, we propose to use the wavelet transform with mediapipe landmarks along with transcriptions and randomly sampled frames to boost the performance of existing SOTA VLMs. The reason why we want to use Mediapipe landmarks is that this gives roughly 400 additional regions mapped directly on faces. The wavelet transform can mimic the trajectory of these subtle facial changes over time. Moreover, they do not take much memory during training and inference time. In this way, we obtain more facial features while maintaining the close memory usage.

**Benchmark**: Source Link: https://arxiv.org/abs/2408.11424. This is the paper I have found that has a similar purpose compared to mine, and the reason I want to use this as the benchmark is that their datasets are high-quality and open for public use as they have human labeled emotion ground truth for each video. Also the average length of these videos are approximately 30 seconds, which means it will work even if we do not have solid GPU resources. 

**Methodology**: We decide to choose Qwen-2.5-VL-3B due to the limited GPU resource we have here (google colab pro with roughly 1000 units from the past). We will not use the full dataset due to the resource limit here. We want to test the zero-shot performance on these datasets, and (2). pre-train the model by adding the wavelet transform and landmark as additional modalities with 20-30% of the full datasets we get and test the performance, and (3). fine tune the model with another 15-20% of the full dataset and test the performance again. Our assumption here is that by giving the additional facial features we are able to perform slightly worse than their models (MAE 10%). 

**Next Step**: We still need to finalize the datasets to be utilized for this project because we believe by applying the wavelet transform we can have up to 100k input videos, which is definitely too much with our limited resources here. Our plan is to cut down the total number of video inputs to 40k at most while maintaining solid quality.




Part 2: Data acquisition and preparation

1. MAFW Source Link: https://dl.acm.org/doi/pdf/10.1145/3503161.3548190 
The open public dataset, MAFW, is comprised of 10045 video clips sourced from diverse media including movies, TV dramas, and short videos across multiple 
cultures (China, Japan, Korea, Europe, America, and India). It provides rich multi-modal annotations including single and multiple
expression labels, bilingual emotional descriptive texts, and automatic annotations such as facial landmarks and gender information. 

Three kinds of annotations are: (1) single expression label; (2) multiple expression label; (3) bilingual emotional descriptive text,
with two subsets: single-expression set, including 11 classes of single emotions; multiple-expression set, including 32 classes of 
multiple emotions, three automatic annotations: the frame-level 68 facial landmarks, bounding boxes of face regions, and gender, 
four benchmarks: uni-modal single expression classification, multi-modal single expression classification, uni-modal compound 
expression classification, and multi-modal compound expression classification. 

The annotations are trustworthy since each video clip is analyzed then given label by 11 trained staff, and this is one of the main reason
that I want to use this dataset. Also, there are clean labels here, which is the other important factor here as we know the better the data
quality the higher chance we can get more robust and better results from models when they make predictions. 

This data set has pre-processed frames for each video clip in 224 x 224 resolution. I plan to use 50% (~5k) of the full dataset here due to 
technlogy limitation and follow a 80-10-10 train/test/val pattern for performance evaluation. 


2. FERV39K Source Link: https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_FERV39k_A_Large-Scale_Multi-Scene_Dataset_for_Facial_Expression_Recognition_in_CVPR_2022_paper.pdf
The open dataset, FERV39K, a large-scale multi-scene dataset, coined as FERV39k. This dataset is novel due to the following
three aspects: (1) multi-scene hierarchy and expression class, (2) generation of candidate video clips, (3) trusted manual
labelling process. I choose 4 scenarios which are subdivided into 22 scenes, annotate 86k samples automatically obtained from 4k videos
based on the well-designed workflow, and finally build 38,935 video clips labeled with 7 classic expressions： “Angry”,
“Disgust”, “Fear”, “Happy”, “Sad”, “Surprise”, “Neutral” are selected as annotation labels.

Labels are made from both crowdsourcing annotator (CA, 20 workers) and professional researcher (PR, 10 workers), respectively. The clips are 
divided into groups at first (5% of each are PR annotated) and copied 3 times. Then they randomly shuffle the grouped materials and provide them 
to CAs. CAs are asked to choose the most likely word or “PASS” on the platform. After annotation, group copies are checked via 
Flag-Recaptured Statistic method. They design 80% and 40% correct rates as two thresholds and mark copies as unacceptable (UA), Improper (IP) and Accept (AC). 
The IP and AC groups will be passed to PRs for judgement, which also made this labeling process reliable. 

I plan to pre-process frames into frames of 224 x 224 resolution. I plan to use 10% (~4k) of the full dataset here due to technlogy limitation 
and follow a 80-10-10 train/test/val pattern for performance evaluation. 

3. DFEW Source Link: https://dfew-dataset.github.io/
Dynamic Facial Expression in-the-Wild (DFEW) is a large-scale facial expression database with 16372 very challenging video clips taken from movies. Clips in the
DFEW database are of various challenging interferences, such as extreme illumination, occlusions, and capricious pose changes. Based on the crowdsourcing annotations,
we hired 12 expert annotators, and each clip has been independently labeled ten times by them. DFEW database has enormous diversities, large quantities, and rich
annotations, including: (1). 16372 number of very challenging video clips from movies, (2). a 7-dimensional expression distribution vector for each video clip,
(3). single-labeled expression annotation for classic seven discrete emotions, (4). baseline classifier outputs based on single-labeled annotation.

As described, the labels are done repeatedly by 12 expert annotators, so they are reasonably to be considered as reliable annotations for training purposes. 
Also it contains lots of variety on video types. 

I plan to pre-process frames into frames of 224 x 224 resolution. I plan to use 30% (~5k) of the full dataset here due to technlogy limitation and 
follow a 80-10-10 train/test/val pattern for performance evaluation. Across the three datasets, I will use ~15k clips for pre-training. For each dataset, 
I follow an 80–10–10 split, where the 20% (val and test) is reserved for evaluation. I will do this project on my own. 


<b></b>Examples in these datasets:

<table id="tfhover" class="tftable" border="1">
<tr><td width="30%"><image src="samples-gif/anger_07317_4s.gif" /></td><td width="15%"><b>Anger</b></td><td>English: A girl with tears in her eyes shouts at the person opposite her. The deep frown,a downward pull on the lip corners,the higher inner corners of eyebrows and the lower outer corners of eyebrows.<br />中文：一个女生眼含着泪水大声训斥着对面的人。眉头紧蹙，嘴角下拉，眉毛内高外低。</td></tr>
<tr><td><image src="samples-gif/disgust_07734.gif" /></td><td><b>Disgust</b></td><td>English: A woman looks nervously at her feet. The frown,the closed eyes and  the  open mouth.<br />中文：一个女人紧张的看着脚下的东西。皱眉，眼睛微闭，嘴巴张开。</td></tr>
<tr><td><image src="samples-gif/fear_09246.gif" /></td><td><b>Fear</b></td><td>English: A girl gasps in the dark. The wide eyes and the open mouth.<br />中文：一个女孩在昏暗的环境中急促的喘息。瞪眼，嘴巴张大。</td></tr>
<tr><td><image src="samples-gif/happy_01440.gif" /></td><td><b>Happiness</b></td><td>English: A woman communicates with a man, talking about dinner. The slightly closed eyes, the open mouth and the raised lip corners.<br />中文：一个女人与男人交流，谈论着晚餐。眼睛微闭，嘴巴张开，嘴角上扬。</td></tr>
<tr><td><image src="samples-gif/sad_00467.gif" /></td><td><b>Sadness</b></td><td>English: A girl stands on the beach, tilting her head back and crying. The deep frown and the wide open mouth.<br />中文：一个女孩站在海边，仰着头哭泣。眉头紧蹙，嘴巴张大。</td></tr></table>

<b></b>Categories expect to see within these datasets:

<tr><td width="30%"><image src="samples-gif/example_category.png" /></td><td width="15%"></td></tr>


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

Part 4: Final Solution

**4.1 Overview**
By this stage, the entire pipeline for the project is implemented and running end-to-end. All components—including preprocessing, segmentation, MediaPipe landmark extraction, blendshape extraction, wavelet-based temporal compression, and classification—have been integrated into a unified system. I made some changes to the data structure of DFEW since originally they do not have scene annotations, which then I used Qwen2.5-VL-3B to generate by feeding some human annotated samples from MAFW. The reason I chose Qwen2.5-VL instead of Qwen3-vl was that Qwen2.5-VL has a default way to read in videos and then extract frames from them, which has been verified with excellent performances, whereas Qwen3-vl requires users to extract frames first and then pass them as image inputs, which may lead to bias. I totally have 8034 samples from MAFW dataset, including 6427 and 1607 train/val samples (80-20 split) with true emotion label and generated annotations. For DFEW, I have a total of 11697 samples, including 9356 and 2341 train/val samples (80-20 split) with true emotion label and human-labeled annotation.  

**4.2 Justification of the Classifier Choice**

After completing preprocessing, segmentation, and feature extraction, the final step in the pipeline is the classification model that maps wavelet-compressed features to one of the seven emotion categories. I selected a lightweight Multi-Layer Perceptron (MLP) classifier for the following reasons: 
(1) Structure of the Input Features: The extracted features consist of:
    - Landmark trajectories: (10, 478, 3) wavelet coefficients
    - Blendshape trajectories: (10, 52) wavelet coefficients
These features are: Low-dimensional compared to raw video; Temporally aligned; Already compressed to multi-scale wavelet descriptors; and Continuous numeric vectors, which means the classifier does not need a heavy temporal model (e.g., LSTM, 3D CNN, or Transformer). Temporal dynamics have already been encoded into fixed-length vectors.

(2) Why is MLP the Appropriate choice: A simple MLP is well-suited because Wavelet transform already captures multi-scale temporal structure, reducing the burden on the classifier. Landmark + blendshape features are dense, continuous, and structured, which MLPs handle efficiently. In addition, MLPs avoid overfitting better than deeper CNNs or transformers when training data is limited.

In conclusion, the MLP strikes the right balance: expressive enough to learn meaningful distinctions, but simple enough to avoid memorizing person-specific geometric patterns.


**4.3 Classification Accuracy: Training vs. Validation**

The classifier was trained on the wavelet-based landmark + blendshape representations of the DFEW sample dataset. Performance is measured using simple classification accuracy.

**4.4 Commentary on Observed Accuracy and Model Behavior**


**4.5 Instructions for Running the Final Code**
(1). Landmark + Blendshape Extraction: This is one of the most important part because we want to get these features from videos. By doing so, you need to change directory to "paligema-mlp" and then run this command: "python3 preprocess_landmarks.py". Make sure you change these paths: "csv_path" and "output_path". 
(2). Run command "python3 train_paligemma.py", which means you are now training a small portion of the paligemma model. Make sure you changes these paths: "model_path", "train_csv", "landmark_path", "output_dir", "checkpoint_path". 

