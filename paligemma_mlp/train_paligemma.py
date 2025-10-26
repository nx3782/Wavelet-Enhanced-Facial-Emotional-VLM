import os
import logging
import wandb
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from math import ceil, sqrt
import gc

from modeling_paligemma import PaliGemmaForConditionalGeneration
from modeling_paligemma import PaliGemmaForRegression
from processing_paligemma import PaliGemmaProcessor
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoLandmarkDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        landmark_path: str,
        processor: PaliGemmaProcessor,
    ):
        self.data = pd.read_csv(csv_path).dropna()
        self.landmarks = np.load(landmark_path)
        self.processor = processor

        if 'promis_anx' not in self.data.columns or 'promis_dep' not in self.data.columns:
            raise ValueError("CSV must contain 'promis_anx' and 'promis_dep' columns for regression.")

        
        if len(self.landmarks.shape) != 4 or self.landmarks.shape[1:] != (10, 478, 3):
            raise ValueError(
                f"Landmark array must have shape (N, 10, 478, 3), got {self.landmarks.shape}"
            )
        
        if 'video_idx' not in self.data.columns:
            raise ValueError("CSV must contain a 'video_idx' column")
            
        if self.data['video_idx'].max() >= len(self.landmarks):
            raise ValueError(
                f"video_idx in CSV exceeds landmark array size. Max index: {self.data['video_idx'].max()}, "
                f"Landmark array size: {len(self.landmarks)}"
            )

    def load_grid_image(self, image_path: str) -> Image.Image:
        """Load a pre-processed grid image from disk."""
        try:
            return Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Error loading image from {image_path}: {str(e)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        image_path = row['file_path']  # New column name for pre-processed grid images
        prompt = row['prompt']
        target1 = row['promis_anx']
        target2 = row['promis_dep']
        targets = torch.tensor([target1, target2], dtype=torch.bfloat16)
        landmark_data = torch.tensor(self.landmarks[row['video_idx']], dtype=torch.bfloat16)

        grid_image = self.load_grid_image(image_path)
        
        return {
            "image": grid_image,
            "prompt": prompt,
            "targets": targets,
            "landmarks": landmark_data
        }


class PaliGemmaDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        landmark_path: str,
        processor: PaliGemmaProcessor,
        batch_size: int = 2,
        num_workers: int = 4,
        train_val_split: float = 0.9
    ):
        super().__init__()
        self.train_csv = train_csv
        self.landmark_path = landmark_path
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

    def setup(self, stage=None):
        full_dataset = VideoLandmarkDataset(
            self.train_csv,
            self.landmark_path,
            self.processor
        )
        self.train_dataset = full_dataset
        # train_size = int(self.train_val_split * len(full_dataset))
        # val_size = len(full_dataset) - train_size
        # self.train_dataset, self.val_dataset = torch.utils.data.random_split(
        #     full_dataset, [train_size, val_size]
        # )

    def train_collate_fn(self, examples):
        images = [example["image"] for example in examples]
        prompts = [example["prompt"] for example in examples]
        targets = torch.stack([example["targets"] for example in examples])
        landmarks = torch.stack([example["landmarks"] for example in examples])

        inputs = self.processor(
            text=prompts,
            images=images,
            landmarks=landmarks,
            return_tensors="pt",
            padding=True,
        )

        inputs["targets"] = targets

        return inputs

    # def val_collate_fn(self, examples):
    #     images = [example["image"] for example in examples]
    #     prompts = [example["prompt"] for example in examples]
    #     responses = [example["response"] for example in examples]
    #     landmarks = torch.stack([example["landmarks"] for example in examples])

    #     inputs = self.processor(
    #         text=prompts,
    #         images=images,
    #         suffix=responses,
    #         landmarks=landmarks,
    #         return_tensors="pt",
    #         padding=True,
    #         # truncation=True,
    #         # max_length=1024
    #     )
    #     return inputs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_collate_fn,
            shuffle=True,
            num_workers=self.num_workers
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         collate_fn=self.val_collate_fn,
    #         shuffle=False,
    #         num_workers=self.num_workers
    #     )

# -----------------------------
# PaliGemma LightningModule
# -----------------------------
class PaliGemmaLightningModule(L.LightningModule):
    def __init__(self, config, model, processor):
        super().__init__()
        self.config = config
        self.model = model
        self.processor = processor
        self.save_hyperparameters(config)
        

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log("train_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        gc.collect()
        torch.cuda.empty_cache()
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log("val_loss", outputs.loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        gc.collect()
        torch.cuda.empty_cache()
        return outputs.loss

    def configure_optimizers(self):
        # Only LoRA + landmark projector parameters require grad
        # so passing self.parameters() is fine.
        # If you want custom param groups, you can filter them:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        return optimizer


def train_paligemma(
    model_path: str,
    train_csv: str,
    landmark_path: str,
    output_dir: str,
    checkpoint_path: str = None,
    config: dict = None
):
    
    if config is None:
        config = {
            "batch_size": 2,
            "max_epochs": 3,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 32
        }
    
    
        
    tb_logger = TensorBoardLogger(
        save_dir="logs",  # This is the directory where logs will be saved
        name="paligemma-regression"  # This is the subdirectory name for this experiment
    )

    # -------------------------------
    # 1) Load model in bf16
    # -------------------------------
    model = PaliGemmaForRegression.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # device_map="auto"  # optional, if using accelerate or auto device map
    )

    # -----------------------------------------------------
    # 2) Freeze entire PaliGemma except landmark projector
    # -----------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False

    for param in model.landmark_projector.parameters():
        param.requires_grad = True

    # -----------------------------------------------------------------
    # 3) Apply LoRA to the *language model* portion (causal LM) only!
    # -----------------------------------------------------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Wrap the *language model* submodule in LoRA
    from peft import get_peft_model
    model.language_model = get_peft_model(model.language_model, lora_config)

    # Let's see how many trainable params
    # (this should reflect LoRA + landmark projector)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print("Trainable:", name, param.shape)
    

    # Initialize processor
    processor = PaliGemmaProcessor.from_pretrained(model_path)

    # Prepare data
    data_module = PaliGemmaDataModule(
        train_csv=train_csv,
        landmark_path=landmark_path,
        processor=processor,
        batch_size=config["batch_size"]
    )
    
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="paligemma-{step}-{epoch}",
        every_n_train_steps=1000,  # Save every epoch
    )
    
    
    # Wrap the model in a Lightning Module
    model_module = PaliGemmaLightningModule(config, model, processor)
    
    # # Optional early stopping
    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     patience=3,
    #     mode="min"
    # )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=2,                # use all available GPUs
        strategy="ddp",            # distributed data parallel
        max_epochs=config["max_epochs"],
        gradient_clip_val=config["gradient_clip_val"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        precision="bf16-mixed",    # crucial for memory-efficiency
        num_sanity_val_steps=0,
        log_every_n_steps=1
    )
    
    trainer.fit(model_module, data_module, ckpt_path=checkpoint_path if checkpoint_path else None)
    
    # Save final model on global rank 0 only
    if trainer.is_global_zero:
        final_save_path = os.path.join(output_dir, "final_model")
        os.makedirs(final_save_path, exist_ok=True)
        model.save_pretrained(final_save_path)
        processor.save_pretrained(final_save_path)
    


if __name__ == "__main__":
    train_paligemma(
        model_path="google/paligemma2-3b-mix-448",
        train_csv="./data/train_pali_mix_prompt.csv",
        landmark_path="/data/yogesh/celebvtext/processed_wavelet_features_all.npy",
        output_dir="./models/paligemma_regression_finetuned",
        checkpoint_path="./models/paligemma_regression_finetuned/checkpoints/paligemma_regression.ckpt"
    )
