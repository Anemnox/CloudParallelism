#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""Pre-Training a 🤖 Wav2Vec2 model on unlabeled audio data"""

import argparse
import math
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import datasets
import torch
import torch.distributed as dist
from datasets import Dataset,DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import numpy as np

import transformers
from transformers import (
    AdamW,
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    get_scheduler,
    is_wandb_available,
    set_seed,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from transformers.utils import send_example_telemetry

#import smdistributed.dataparallel.torch.torch_smddp  # Import for SageMaker Distributed Data Parallel (SMDP)
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', ''))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', ''))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', ''))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', ''))
    parser.add_argument("--hosts", type=list, default=os.environ.get("SM_HOSTS", []))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST", ''))
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("SM_NUM_GPUS", 1)))
    parser.add_argument("--audio_column_name", type=str, default="audio", help="Name of the audio column.")
    parser.add_argument("--num_nodes", type=int, default=len(os.environ.get("SM_HOSTS", [])))
    parser.add_argument("--epochs",type=int,default=1)
    # Additional arguments for feature extraction, distributed training, etc.
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_duration_in_seconds",
        type=float,
        default=5.0,
        help="Filter out audio files that are longer than `max_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="Type of distributed backend to use",
        choices= ["nccl","smddp"]
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints during training.",
    )
    
    
    args = parser.parse_args()
    
    # Validate gradient accumulation steps
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")
    
    # Calculate world size after parsing arguments
    args.world_size = args.num_gpus * args.num_nodes
    args.batch_size = int(os.environ.get("SM_HP_BATCH_SIZE", 8))
    #args.epochs = int(os.environ.get("SM_HP_EPOCHS", 3))

    # Create necessary directories
    if args.output_data_dir:
        os.makedirs(args.output_data_dir, exist_ok=True)
    # if args.checkpoint_dir:
    #     os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def setup_distributed():
    if "WORLD_SIZE" in os.environ:
        try:
            # SageMaker SMDP environment variables
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            
            # Initialize distributed process group
            dist.init_process_group(
                backend=args.backend,
                rank=rank,
                world_size=world_size
            )
            
            # Set device
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            logger.info(f"Initialized distributed process group with rank {rank}, world size {world_size}, and local rank {local_rank}.")
            if torch.cuda.is_available():
                # Log GPU memory info
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.2f} GB")
                # Enable cuDNN benchmarking for better performance
                torch.backends.cudnn.benchmark = True
        
                    
            return device, local_rank, world_size
        except KeyError as e:
            logger.error(f"Missing environment variable for distributed training: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Failed to initialize distributed process group: {e}")
            raise
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Distributed training not enabled. Using device: %s", device)
        return device, 0, 1


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def get_txt_file_paths(directory):
    txt_file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                txt_file_paths.append(os.path.abspath(os.path.join(root, file)))
    return txt_file_paths


def load_dataset_audio(directory):
    # PREPARE DATA USING FLAC FILES AND LABEL TEXT
    # (rewrite load_dataset method)
    # STEPS
    # 1. GET FOLDER
    # 2. READ TXT IN FOLDER
    # 3. PARSE TXT BY LINE (GET FLAC FILE NAME AND LABEL)
    # 4. MAP EACH FLAC FILE TO LABEL
    # 5. INPUT INTO raw_datasets
    text_files = get_txt_file_paths(directory)
    data = []
    
    for text in text_files:
        absolute_path = os.path.dirname(text)
        with open(text, "r") as f:
            for line in f:
                file_name, label = line.strip().split(maxsplit=1)
                data.append({"audio": f"{absolute_path}/{file_name}.flac", "label": label})

    return Dataset.from_list(data)


def load_datasets_from_s3():
    # Get S3 paths from SageMaker environment variables
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', '')
    val_dir = os.environ.get('SM_CHANNEL_VALIDATION', '')

    if not train_dir or not val_dir:
        raise ValueError("Training or validation data path not provided")

    # Load datasets directly from S3 paths
    raw_datasets = DatasetDict()
    try:
        # Load training dataset
        logger.info(f"Loading training dataset from {train_dir}")
        raw_datasets["train"] = load_dataset_audio(train_dir)

        # Load validation dataset
        logger.info(f"Loading validation dataset from {val_dir}")
        raw_datasets["validation"] = load_dataset_audio(val_dir)

    except Exception as e:
        logger.error(f"Error loading datasets from S3: {e}")
        raise

    logger.info(f"Training dataset size: {len(raw_datasets['train'])}")
    logger.info(f"Validation dataset size: {len(raw_datasets['validation'])}")

    return raw_datasets


def rms_normalize_audio(audio_array, target_level=-25):
        audio_array = audio_array.astype(np.float32)
        rms = np.sqrt(np.mean(audio_array**2))
        if rms > 0:
            target_rms = 10**(target_level/20)
            audio_array = audio_array * (target_rms/rms)
        return audio_array


def prepare_datasets(raw_datasets, feature_extractor, args):
    # Cast audio column with correct sampling rate
    try:
        logger.info("Casting audio column to the correct sampling rate.")
        raw_datasets = raw_datasets.cast_column(
            args.audio_column_name, 
            datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )
    except Exception as e:
        logger.error(f"Error casting audio column: {e}")
        raise

    def prepare_dataset(batch):
        sample = batch[args.audio_column_name]

        audio_array = rms_normalize_audio(sample["array"])

        try:
            inputs = feature_extractor(
                audio_array, 
                sampling_rate=sample["sampling_rate"], 
                max_length=int(args.max_duration_in_seconds * feature_extractor.sampling_rate),
                truncation=True
            )
            batch["input_values"] = inputs.input_values[0]
            batch["input_length"] = len(inputs.input_values[0])
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
            raise
        return batch

    # Set up caching for preprocessed datasets
    try:
        logger.info("Mapping datasets with feature extraction.")
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=True
        )
        logger.info(f"Number of training examples: {len(vectorized_datasets['train'])}")
        logger.info(f"Number of validation examples: {len(vectorized_datasets['validation'])}")
    except Exception as e:
        logger.error(f"Error mapping datasets: {e}")
        raise
    # Filter samples based on length
    min_length = int(args.max_duration_in_seconds * 0.5 * feature_extractor.sampling_rate)
    vectorized_datasets = vectorized_datasets.filter(
        lambda x: x["input_length"] >= min_length,
        num_proc=args.preprocessing_num_workers
    )

    vectorized_datasets = vectorized_datasets.remove_columns("input_length")
    
    return vectorized_datasets


def setup_model_and_data_collator(args, feature_extractor):
    # Load model config
    try:
        logger.info(f"Loading model configuration from {args.model_name_or_path}")
        config = Wav2Vec2Config.from_pretrained(args.model_name_or_path)
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        raise

    # Initialize model
    try:
        logger.info("Initializing the Wav2Vec2 model for pre-training.")
        model = Wav2Vec2ForPreTraining(config)
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

    # Enable gradient checkpointing if specified
    if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
        try:
            logger.info("Enabling gradient checkpointing for the model.")
            model.gradient_checkpointing_enable()
        except Exception as e:
            logger.error(f"Error enabling gradient checkpointing: {e}")
            raise

    # Move model to GPU and wrap in SMDP if distributed
    try:
        device, local_rank, _ = setup_distributed()
        model = model.to(device)
    except Exception as e:
        logger.error(f"Error moving model to device: {e}")
        raise
    
    if is_distributed():
        try:
            logger.info("Wrapping model in DistributedDataParallel.")
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank
            )
        except Exception as e:
            logger.error(f"Error wrapping model in DistributedDataParallel: {e}")
            raise

    # Setup masking parameters for data collator
    mask_time_prob = config.mask_time_prob if hasattr(config, 'mask_time_prob') else 0.065
    mask_time_length = config.mask_time_length if hasattr(config, 'mask_time_length') else 10

    # Create data collator
    try:
        logger.info("Creating data collator for Wav2Vec2 pretraining.")
        data_collator = DataCollatorForWav2Vec2Pretraining(
            model=model,
            feature_extractor=feature_extractor,
            pad_to_multiple_of=8,
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length,
        )
    except Exception as e:
        logger.error(f"Error creating data collator: {e}")
        raise

    return model, data_collator, config


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.
    """

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.065
    mask_time_length: Optional[int] = 10

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Reformat list to dict and set to pytorch format
        try:
            logger.debug("Padding batch of features.")
            batch = self.feature_extractor.pad(
                features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
        except Exception as e:
            logger.error(f"Error during padding of features: {e}")
            raise

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        try:
            mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
            mask_indices_seq_length = int(mask_indices_seq_length)

            # Make sure that no loss is computed on padded inputs
            if batch.get("attention_mask") is not None:
                logger.debug("Computing sub-attention mask for padded inputs.")
                batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                    mask_indices_seq_length, batch["attention_mask"]
                )

            features_shape = (batch_size, mask_indices_seq_length)

            # Sample randomly masked indices
            logger.debug("Sampling masked time indices.")
            mask_time_indices = _compute_mask_indices(
                features_shape,
                self.mask_time_prob,
                self.mask_time_length,
                attention_mask=batch.get("sub_attention_mask"),
            )

            # Sample negative indices
            logger.debug("Sampling negative indices for contrastive learning.")
            sampled_negative_indices = _sample_negative_indices(
                features_shape,
                self.model.config.num_negatives,
                mask_time_indices=mask_time_indices,
            )
            batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
            batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)
            logger.debug(f"Batch size: {batch_size}")
            logger.debug(f"Mask indices sequence length: {mask_indices_seq_length}")
            logger.debug(f"Features shape: {features_shape}")
        except Exception as e:
            logger.error(f"Error during data collation: {e}")
            raise

        return batch


def main(args):
    send_example_telemetry("run_wav2vec2_pretraining_no_trainer", args)

    # Set the training seed now.
    if args.seed is not None:
        logger.info(f"Setting seed for reproducibility: {args.seed}")
        set_seed(args.seed)
    
    # Load dataset
    logger.info("Loading datasets from S3.")
    raw_datasets = load_datasets_from_s3()
    
    # Initialize feature extractor
    try:
        logger.info(f"Loading feature extractor from {args.model_name_or_path}")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
    except Exception as e:
        logger.error(f"Error loading feature extractor: {e}")
        raise
    
    # Prepare datasets
    logger.info("Preparing datasets for training.")
    vectorized_datasets = prepare_datasets(raw_datasets, feature_extractor, args)

    # Setup model and data collator
    logger.info("Setting up model and data collator.")
    model, data_collator, config = setup_model_and_data_collator(args, feature_extractor)
    
    # Create DataLoaders with DistributedSampler for distributed training
    try:
        train_sampler = DistributedSampler(vectorized_datasets["train"]) if "WORLD_SIZE" in os.environ else None
        train_dataloader = DataLoader(
            vectorized_datasets["train"],
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.preprocessing_num_workers,
            pin_memory=True,
            drop_last=True  # Important for distributed training
        )

        eval_sampler = DistributedSampler(vectorized_datasets["validation"]) if "WORLD_SIZE" in os.environ else None
        eval_dataloader = DataLoader(
            vectorized_datasets["validation"],
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            sampler=eval_sampler,
        )
    except Exception as e:
        logger.error(f"Error creating DataLoaders: {e}")
        raise

    # Setup optimizer and learning rate scheduler
    logger.info("Setting up optimizer and learning rate scheduler.")
    try:
        optimizer = AdamW(model.parameters(), lr=3e-4)
        num_training_steps = args.epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
    except Exception as e:
        logger.error(f"Error setting up optimizer or learning rate scheduler: {e}")
        raise

    # Mixed Precision Training
    scaler = GradScaler()
    gumbel_temperature = config.max_gumbel_temperature
    # Training Loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(args.epochs):
        model.train()
        logger.info(f"Starting training for epoch {epoch}.")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                with autocast():  # Mixed precision
                    outputs = model(**batch)
                    loss = outputs.loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()
                logger.info(f"Loss at step {step}: {loss.item()}")
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                # Update Gumbel temperature
                gumbel_temperature = max(
                    config.max_gumbel_temperature * args.gumbel_temperature_decay ** (step + epoch * len(train_dataloader)),
                    config.min_gumbel_temperature
                )

                if hasattr(model, 'module'):
                    model.module.set_gumbel_temperature(gumbel_temperature)
                else:
                    model.set_gumbel_temperature(gumbel_temperature)
            except Exception as e:
                logger.error(f"Error during training step {step} of epoch {epoch}: {e}")
                raise

        # Save checkpoint
        # checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        # logger.info(f"Saving checkpoint to {checkpoint_path}")
        # try:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': lr_scheduler.state_dict(),
        #         'scaler_state_dict': scaler.state_dict()
        #     }, checkpoint_path)
        # except Exception as e:
        #     logger.error(f"Error saving checkpoint at epoch {epoch}: {e}")
        #     raise

        # Evaluation Loop
        # logger.info(f"Starting evaluation for epoch {epoch}.")
        # model.eval()
        # eval_loss = 0
        # try:
        #     for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
        #         batch = {k: v.to(device) for k, v in batch.items()}
        #         with torch.no_grad():
        #             outputs = model(**batch)
        #             eval_loss += outputs.loss.item()
        #     eval_loss /= len(eval_dataloader)
        #     logger.info(f"Epoch {epoch}: Evaluation Loss: {eval_loss}")
        # except Exception as e:
        #     logger.error(f"Error during evaluation at epoch {epoch}: {e}")
        #     raise

    cleanup_distributed()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting main training script.")
    try:
        main(args)
    except Exception as e:
        logger.error(f"Fatal error in main script: {e}")
        raise
