import os
import torch
import lightning as L
from torch.utils.data import DataLoader

from modeling_paligemma import PaliGemmaForConditionalGeneration
from modeling_paligemma import PaliGemmaForRegression
from processing_paligemma import PaliGemmaProcessor
from train_paligemma import (
    PaliGemmaLightningModule, 
    VideoLandmarkDataset, 
    PaliGemmaDataModule
)
from peft import LoraConfig, get_peft_model, TaskType

def continue_training():
    # Paths and configuration
    checkpoint_path = "./models/paligemma-step=3000-epoch=1.ckpt"
    base_model_path = "google/paligemma2-3b-mix-448"
    train_csv = "./data/emotion_recognition_one_prompt_better_trans.csv"
    landmark_path = "./data/combined_dataset/wavelets_train.npy"
    output_dir = "./models/paligemma_landmark_continued"
    
    # Configuration (same as original training)
    config = {
            "batch_size": 2,
            "max_epochs": 3,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 32
        }
    
    # 1) Load base model
    base_model = PaliGemmaForRegression.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
    )
    
    # 2) Freeze most parameters except landmark projector
    for param in base_model.parameters():
        param.requires_grad = False
        
    for param in base_model.landmark_projector.parameters():
        param.requires_grad = True
        
    for param in base_model.multi_modal_projector.parameters():
        param.requires_grad = True
    
    # 3) Setup LoRA for language model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA configuration
    base_model.language_model = get_peft_model(base_model.language_model, lora_config)
    
    # 4) Load processor
    processor = PaliGemmaProcessor.from_pretrained(base_model_path)
    
    # 5) Create Lightning module with the model
    model_module = PaliGemmaLightningModule(config, base_model, processor)
    
    # # 6) Load checkpoint with strict=False
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_module.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # model_module = PaliGemmaLightningModule.load_from_checkpoint(
    #     checkpoint_path,
    #     model=base_model,
    #     processor=processor
    # )
    
    # Check which weights were loaded vs. missing
    missing_keys = set(model_module.state_dict().keys()) - set(checkpoint['state_dict'].keys())
    print(f"Missing keys in checkpoint: {len(missing_keys)}")
    
    
    # 7) Setup data module for training
    data_module = PaliGemmaDataModule(
        train_csv=train_csv,
        landmark_path=landmark_path,
        processor=processor,
        batch_size=config["batch_size"]
    )
    
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "epoch_checkpoints"), exist_ok=True)
    
    # 8) Setup checkpoint callback
    # step_checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    #     dirpath=os.path.join(output_dir, "checkpoints"),
    #     filename="paligemma-continued-{step}-{epoch}",
    #     every_n_train_steps=100,
    # )

    epoch_checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(output_dir, "epoch_checkpoints"),
        filename="paligemma-continued-epoch{epoch}",
        save_on_train_epoch_end=True,  
        every_n_epochs=1, 
    )
    
    # 9) Setup logger
    tb_logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir="logs",
        name="paligemma-landmark-continued"
    )
    
    # 10) Create trainer and continue training
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="ddp",
        max_epochs=config["max_epochs"],
        gradient_clip_val=config["gradient_clip_val"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=[epoch_checkpoint_callback],#[step_checkpoint_callback, epoch_checkpoint_callback],
        logger=tb_logger,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        log_every_n_steps=1
    )
    
    # Begin training from the checkpoint
    trainer.fit(model_module, data_module)

if __name__ == "__main__":
    continue_training()