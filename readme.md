Part 1: Conceptual_design

**Abstract**: With the fast development of Large Language Models (LLMs), people have lots of mature and powerful AI tools to help them not only solve daily matters but also problems that are really challenging; for example, a series of mathematical problems that the latter problems one depend on the answers of the previous ones and step by step mathematical proofs that require strong logics. With such powerful LLMs as great backbones, computer scientists start to shift their focus more on Vision-Language models (VLMs) and also Multi-modal Models (MMs) to help machines get more useful information to better assist humans in all kinds of situations.

**Problems**: Extracting features from vision inputs has been one of the greatest challenges for these models, especially long video sequence inputs due to the fact that most state-of-the-art (SOTA) models still random sample frames from each second of the video input to reduce memory burden, which leads to unavoidable information loss, and simply selecting more frames will exponentially increase the training and inference time. 

**Solution**: To solve this problem, we propose to use the wavelet transform with mediapipe landmarks along with transcriptions and randomly sampled frames to boost the performance of existing SOTA VLMs. The reason why we want to use Mediapipe landmarks is that this gives roughly 400 additional regions mapped directly on faces. The wavelet transform can mimic the trajectory of these subtle facial changes over time. Moreover, they do not take much memory during training and inference time. In this way, we obtain more facial features while maintaining the close memory usage.

**Benchmark**: https://arxiv.org/abs/2408.11424. This is the paper I have found that has a similar purpose compared to mine, and the reason I want to use this as the benchmark is that their datasets are high-quality and open for public use as they have human labeled emotion ground truth for each video. Also the average length of these videos are approximately 30 seconds, which means it will work even if we do not have solid GPU resources. 

**Methodology**: We decide to choose Qwen-2.5-VL-3B due to the limited GPU resource we have here (google colab pro with roughly 1000 units from the past). We will not use the full dataset due to the resource limit here. We want to test the zero-shot performance on these datasets, and (2). pre-train the model by adding the wavelet transform and landmark as additional modalities with 20-30% of the full datasets we get and test the performance, and (3). fine tune the model with another 15-20% of the full dataset and test the performance again. Our assumption here is that by giving the additional facial features we are able to perform slightly worse than their models (MAE 10%). 

**Next Step**: We still need to finalize the datasets to be utilized for this project because we believe by applying the wavelet transform we can have up to 100k input videos, which is definitely too much with our limited resources here. Our plan is to cut down the total number of video inputs to 40k at most while maintaining solid quality.




Part 2: Data acquisition and preparation

1. MAFW: https://dl.acm.org/doi/pdf/10.1145/3503161.3548190 
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


2. FERV39K: https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_FERV39k_A_Large-Scale_Multi-Scene_Dataset_for_Facial_Expression_Recognition_in_CVPR_2022_paper.pdf
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

3. DFEW, link: https://dfew-dataset.github.io/
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
