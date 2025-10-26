import os
import re
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
from sklearn.metrics import recall_score, confusion_matrix, classification_report
import random
from modeling_paligemma import PaliGemmaForConditionalGeneration
from processing_paligemma import PaliGemmaProcessor
from train_paligemma import PaliGemmaLightningModule
from peft import LoraConfig, get_peft_model, TaskType

def load_model_and_processor(base_model_path, checkpoint_path):
    """
    Load the PaliGemma model with landmark projector and LoRA weights.
    """
    print(f"Loading base model from {base_model_path}")
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16
    )
    
    # Configure LoRA (same configuration as in training)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA to language model component
    base_model.language_model = get_peft_model(base_model.language_model, lora_config)
    
    # Load processor
    processor = PaliGemmaProcessor.from_pretrained(base_model_path)
    
    print(f"Loading trained weights from checkpoint: {checkpoint_path}")
    
    # Load the full model from checkpoint
    lightning_module = PaliGemmaLightningModule.load_from_checkpoint(
        checkpoint_path,
        model=base_model,
        processor=processor,
        map_location="cpu"
    )
    
    # Put model in evaluation mode
    lightning_module.eval()
    
    return lightning_module, processor


def extract_emotion_label(generated_text):
    """
    Extract emotion label from the generated text using regex pattern matching.
    The function looks for text between tags or explicit emotion mentions.
    """
    # Look for text between <emotion> and </emotion> tags
    pattern = r'<emotion>(.*?)</emotion>'
    matches = re.search(pattern, generated_text, re.IGNORECASE | re.DOTALL)
    
    if matches:
        # Extract and clean the text between the tags
        emotion = matches.group(1).strip().lower()
        return emotion
    return "unknown"


def evaluate_test_set(
    test_csv_path, 
    landmark_path, 
    base_model_path, 
    checkpoint_path, 
    output_csv_path,
    batch_size=8  # Increased default batch size
):
    """
    Run batched inference on test set and evaluate performance.
    
    Args:
        test_csv_path: Path to test CSV file
        landmark_path: Path to numpy file with landmark features
        base_model_path: Path to base PaliGemma model
        checkpoint_path: Path to trained checkpoint
        output_csv_path: Path to save results CSV
        batch_size: Batch size for inference
    """
    # df_train = pd.read_csv("./data/emotion_recognition_one_prompt.csv")
    df_train = pd.read_csv("./data/emotion_recognition_one_prompt_better_trans.csv")
    
    # Load model and processor
    lightning_module, processor = load_model_and_processor(base_model_path, checkpoint_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_module.to(device)
    print(f"Using device: {device}")
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    # dataset_name = test_df['file_path'][0].split("/")[2]
    # prompt = list(df_train[df_train['dataset']==dataset_name]["prompt"].unique())[0]
    
    print(f"Loaded test set with {len(test_df)} samples")
    
    # Load landmark features
    landmarks = np.load(landmark_path)
    print(f"Loaded landmark features with shape: {landmarks.shape}")
    
    # Prepare result storage
    results = []
    true_labels = []
    pred_labels = []
    
    # Process data in batches
    print(f"Starting inference with batch size: {batch_size}")
    start_time = time.time()
    
    # Create batches
    num_samples = len(test_df)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_idx in tqdm(range(num_batches)):
        # Get batch indices
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_size_actual = end_idx - start_idx
        
        # Initialize batch containers
        batch_images = []
        batch_prompts = []
        batch_landmarks = []
        batch_true_labels = []
        batch_file_paths = []
        
        # Collect batch data
        for idx in range(start_idx, end_idx):
            try:
                row = test_df.iloc[idx]
                
                # Load image
                image_path = row['file_path']
                batch_file_paths.append(image_path)
                image = Image.open(image_path)
                batch_images.append(image)
                
                # Create prompt
                # labels = "anger, disgust, fear, happiness, sadness, surprise, contempt, anxiety, helplessness, disappointment, neutral"
                labels = "happy, sad, neutral, angry, surprise, disgust, fear"
                transcription = row['transcription']
                prompt = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the overall emotion. The reasoning process and classified emotion label are enclosed within <think> </think> and <emotion> </emotion> tags, respectively, i.e., <think> reasoning process here </think> <emotion> emotion here </emotion>. User: Transcription: \"{transcription}\"\nDetermine the primary emotional state of the person in the image by analyzing their facial expressions over time. Make sure to utilize the wavelets of the facial landmark trajectories along with the video frames in the image, as well as the provided transcription. Classify the emotion into one of the following categories: {labels}. Assistant:\n"
                # prompts = list(df_train[df_train['file_path']==image_path]["prompt"].unique())
                # prompt = prompts[0]
                # print(prompt)
                # prompt=row['prompt']
                batch_prompts.append(prompt)
                
                # Get ground truth label
                true_label = row['label_name'].strip().lower() if 'label_name' in row else "unknown"
                batch_true_labels.append(true_label)
                
                # Get landmark data
                video_idx = row['video_idx']
                landmark_data = landmarks[video_idx]
                batch_landmarks.append(landmark_data)
                
            except Exception as e:
                print(f"Error preparing sample {idx}: {e}")
                # If there's an error, reduce the batch size and skip this sample
                batch_size_actual -= 1
                continue
        
        # If all samples in the batch failed, continue to the next batch
        if batch_size_actual == 0:
            continue
            
        try:
            # Convert landmarks to tensor and ensure proper shape
            batch_landmarks_tensor = torch.tensor(np.array(batch_landmarks), dtype=torch.bfloat16)
            print("Landmarks:", batch_landmarks_tensor.shape)
            
            # Process batch input
            inputs = processor(
                text=batch_prompts,
                images=batch_images,
                landmarks=batch_landmarks_tensor,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            
            # Record input length for trimming generated output
            input_len = inputs["input_ids"].shape[1]
            
            # Generate output for the entire batch
            with torch.inference_mode():
                generations = lightning_module.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False
                )
                
            # Process each generated output in the batch
            for i in range(batch_size_actual):
                # Only keep the generated part (not input)
                generated_tokens = generations[i][input_len:]
                generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
                # print(generated_text)
                # Extract emotion label
                pred_label = extract_emotion_label(generated_text)
                # pred_label = generated_text
                
                # Store results
                true_labels.append(batch_true_labels[i])
                pred_labels.append(pred_label)
                
                # print("Generated Text:", generated_text)
                print("Predicted:", pred_label)
                print("Actual:", batch_true_labels[i])
                
                results.append({
                    'file_path': batch_file_paths[i],
                    'true_label': batch_true_labels[i],
                    'predicted_label': pred_label,
                    'generated_text': generated_text,
                    'correct': batch_true_labels[i] == pred_label
                })
            
            # Clean up to prevent memory buildup
            del batch_images, batch_landmarks, batch_landmarks_tensor, inputs, generations
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Add error entries for all samples in this batch
            for i in range(batch_size_actual):
                results.append({
                    'file_path': batch_file_paths[i],
                    'true_label': batch_true_labels[i],
                    'predicted_label': "error",
                    'generated_text': str(e),
                    'correct': False
                })
    
    # Create results DataFrame and calculate metrics
    # (rest of the function remains unchanged)
    results_df = pd.DataFrame(results)
    
    results_df = results_df[results_df['predicted_label']!='unknown']
    
    # Calculate accuracy
    accuracy = results_df['correct'].mean()
    print(f"Accuracy: {accuracy:.4f}")
    
    labels = list(test_df['label_name'].unique())
    
    # Calculate unweighted recall (macro average)
    macro_recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0, labels=labels)
    print(f"Unweighted Average Recall (Macro): {macro_recall:.4f}")
    
    # Calculate weighted recall
    weighted_recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0, labels=labels)
    print(f"Weighted Average Recall: {weighted_recall:.4f}")
    
    
    # Add evaluation metrics to the DataFrame
    metrics_df = pd.DataFrame([{
        'accuracy': accuracy,
        'macro_recall': macro_recall,
        'weighted_recall': weighted_recall,
        'total_samples': len(results_df),
        'batch_size': batch_size,
        'inference_time': time.time() - start_time
    }])
    
    # Save results to CSV
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
    
    # Save metrics to a separate CSV
    metrics_path = output_csv_path.replace('.csv', '_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    
    return results_df, accuracy, macro_recall, weighted_recall


if __name__ == "__main__":
    # Configuration parameters
    base_model_path = "google/paligemma2-3b-mix-448"  # Path to base model
    checkpoint_path = "./models/cot-epoch1.ckpt"  # Path to trained checkpoint
    test_csv_path = "./data/DFEW/dfew_test_trans.csv"  # Path to test CSV
    landmark_path = "./data/DFEW/wavelets_test.npy"  # Path to landmark features
    output_csv_path = "./results/dfew_results_trans_cot.csv"  # Path to save results
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Run evaluation
    results_df, accuracy, macro_recall, weighted_recall = evaluate_test_set(
        test_csv_path=test_csv_path,
        landmark_path=landmark_path,
        base_model_path=base_model_path,
        checkpoint_path=checkpoint_path,
        output_csv_path=output_csv_path
    )
    
    print("\nEvaluation complete!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Unweighted Average Recall (Macro): {macro_recall:.4f}")
    print(f"Weighted Average Recall: {weighted_recall:.4f}")