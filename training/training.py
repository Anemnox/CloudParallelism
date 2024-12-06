#!/usr/bin/env python
# coding=utf-8
import argparse
import math
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
import torch.distributed as dist
import numpy as np
import datasets
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import time
import transformers
from transformers import (
    AdamW,
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    get_scheduler,
    set_seed,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
import smdistributed.dataparallel.torch.torch_smddp as smddp
from torch.nn.parallel import DistributedDataParallel as DDP
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
    parser.add_argument("--epochs", type=int, default=1)
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
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints during training.",
    )
    parser.add_argument(
        "--max_gumbel_temperature",
        type=float,
        default=2.0,
        help="Maximum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--min_gumbel_temperature",
        type=float,
        default=0.5,
        help="Minimum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--gumbel_temperature_decay", type=float, default=0.999995, help="Decay of gumbel temperature during training."
    )
    args = parser.parse_args()

    # Validate gradient accumulation steps
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")
    
    # Calculate world size after parsing arguments
    args.world_size = args.num_gpus * args.num_nodes
    args.batch_size = int(os.environ.get("SM_HP_BATCH_SIZE", 8))

    # Create necessary directories
    if args.output_data_dir:
        os.makedirs(args.output_data_dir, exist_ok=True)

    return args


def setup_distributed():
    try:
        if smddp.is_available:
            logger.info("Initializing SageMaker Distributed Data Parallel (SMDP).")
            dist.init_process_group(backend="smddp")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = rank % torch.cuda.device_count()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            logger.info(f"Initialized distributed process group with rank {rank}, world size {world_size}, and local rank {local_rank}.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            rank = 0
            world_size = 1
            local_rank = 0
            logger.info("Distributed training not enabled. Using device: %s", device)
        
        return device, local_rank, world_size
    except RuntimeError as e:
        logger.error(f"Failed to initialize distributed process group: {e}")
        raise


def is_distributed():
    return smddp.is_available and dist.get_world_size() > 1


def cleanup_distributed():
    torch.cuda.empty_cache()
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
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', '')
    val_dir = os.environ.get('SM_CHANNEL_VALIDATION', '')

    if not train_dir or not val_dir:
        raise ValueError("Training or validation data path not provided")

    raw_datasets = DatasetDict()
    try:
        logger.info(f"Loading training dataset from {train_dir}")
        raw_datasets["train"] = load_dataset_audio(train_dir)

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
                max_length=int(10 * feature_extractor.sampling_rate),  # Set max_length here
                truncation=True
            )
            batch["input_values"] = inputs.input_values[0]
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
            raise
        return batch

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

    return vectorized_datasets


def setup_model_and_data_collator(args, feature_extractor):
    try:
        device, local_rank, world_size = setup_distributed()
        logger.info(f"Distributed setup complete. Device: {device}, Local rank: {local_rank}, World size: {world_size}")
    except Exception as e:
        logger.error(f"Error setting up distributed environment: {e}")
        raise

    try:
        logger.info(f"Loading model configuration from {args.model_name_or_path}")
        config = Wav2Vec2Config.from_pretrained(args.model_name_or_path)
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        raise

    try:
        logger.info("Initializing the Wav2Vec2 model for pre-training.")
        model = Wav2Vec2ForPreTraining(config)
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Gradient checkpointing not supported by this model")
        
        if is_distributed():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            logger.info("Model wrapped in DistributedDataParallel")
            
    except Exception as e:
        logger.error(f"Error setting up model: {e}")
        raise

    mask_time_prob = getattr(config, 'mask_time_prob', 0.065)
    mask_time_length = getattr(config, 'mask_time_length', 10)
    logger.info(f"Using mask_time_prob={mask_time_prob}, mask_time_length={mask_time_length}")

    try:
        logger.info("Creating data collator for Wav2Vec2 pretraining.")
        data_collator = DataCollatorForWav2Vec2Pretraining(
            model=model,
            feature_extractor=feature_extractor,
            pad_to_multiple_of=8,
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length,
        )
        logger.info("Data collator created successfully.")
    except Exception as e:
        logger.error(f"Error creating data collator: {e}")
        raise

    return model, data_collator, config, device

@dataclass
class DataCollatorForWav2Vec2Pretraining:
    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.065
    mask_time_length: Optional[int] = 10

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        try:
            # Make sure the padding is done to max_length
            batch = self.feature_extractor.pad(
                features,
                padding=self.padding,  # Use max_length to ensure consistency
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            # Ensure mask indices are computed based on the consistent max_length
            batch_size = batch["input_values"].shape[0]
            actual_model = self.model.module if hasattr(self.model, "module") else self.model

            mask_indices_seq_length = actual_model._get_feat_extract_output_lengths(
                batch["input_values"].shape[-1]
            )
            mask_indices_seq_length = int(mask_indices_seq_length)

            if mask_indices_seq_length <= 0:
                raise ValueError(f"Invalid mask indices sequence length: {mask_indices_seq_length}")

            if "attention_mask" in batch:
                batch["sub_attention_mask"] = actual_model._get_feature_vector_attention_mask(
                    mask_indices_seq_length,
                    batch["attention_mask"]
                )

            features_shape = (batch_size, mask_indices_seq_length)

            # Compute mask indices based on the fixed max_length shape
            mask_time_indices = _compute_mask_indices(
                features_shape,
                self.mask_time_prob,
                self.mask_time_length,
                attention_mask=batch.get("sub_attention_mask")
            )

            negative_count = actual_model.config.num_negatives
            sampled_negative_indices = _sample_negative_indices(
                features_shape,
                negative_count,
                mask_time_indices=mask_time_indices
            )

            # Convert mask indices and sampled negative indices to tensors
            batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long)
            batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long)

            return batch

        except Exception as e:
            logger.error(f"Error in data collation: {str(e)}")
            raise


def main(args):
    if args.seed is not None:
        logger.info(f"Setting seed for reproducibility: {args.seed}")
        set_seed(args.seed)
    
    logger.info("Loading datasets from S3.")
    raw_datasets = load_datasets_from_s3()
    
    try:
        logger.info(f"Loading feature extractor from {args.model_name_or_path}")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
    except Exception as e:
        logger.error(f"Error loading feature extractor: {e}")
        raise
    
    logger.info("Preparing datasets for training.")
    vectorized_datasets = prepare_datasets(raw_datasets, feature_extractor, args)

    logger.info("Setting up model and data collator.")
    model, data_collator, config, device = setup_model_and_data_collator(args, feature_extractor)
    
    try:
        train_sampler = DistributedSampler(
            vectorized_datasets['train'],
            num_replicas=dist.get_world_size() if is_distributed() else 1,
            rank=dist.get_rank() if is_distributed() else 0
        )
        train_dataloader = DataLoader(
            vectorized_datasets["train"],
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.preprocessing_num_workers,
            pin_memory=True,
            drop_last=True
        )
    except Exception as e:
        logger.error(f"Error creating DataLoaders: {e}")
        raise

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

    scaler = GradScaler()
    completed_steps = 0
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_network_latency = 0
        num_network_operations = 0
        num_steps = 0

        model.train()
        logger.info(f"Starting training for epoch {epoch}.")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        iterator = train_dataloader
        if not is_distributed() or dist.get_rank() == 0:
            iterator = tqdm(iterator, desc=f"Training Epoch {epoch}")
        
        for step, batch in enumerate(iterator):
            try:
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                network_start_time = time.time()
                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                completed_steps += 1
                num_steps += 1
                network_latency = time.time() - network_start_time
                total_network_latency += network_latency
                num_network_operations += 1
                if step % 100 == 0 and is_distributed():
                    torch.cuda.synchronize()
            except Exception as e:
                logger.error(f"Error during training step {step} of epoch {epoch}: {e}")
                if is_distributed():
                    cleanup_distributed()
                raise
        if is_distributed():
            torch.distributed.barrier()
        epoch_duration = time.time() - epoch_start_time
        throughput = len(train_dataloader.dataset) / epoch_duration
        avg_network_latency = total_network_latency / num_network_operations if num_network_operations > 0 else 0
        if not is_distributed() or dist.get_rank() == 0:
            logger.info(f"Epoch {epoch + 1} training_time: {epoch_duration:.2f} seconds")
            logger.info(f"Epoch {epoch + 1} throughput: {throughput:.2f} samples/second")
            logger.info(f"Epoch {epoch + 1} avg_network_latency: {avg_network_latency:.2f} seconds")
    cleanup_distributed()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting main training script.")
    try:
        main(args)
    except Exception as e:
        logger.error(f"Fatal error in main script: {e}")
        raise
