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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training
try:
    backend = "smddp"
    import smdistributed.dataparallel.torch.torch_smddp as smddp
except ModuleNotFoundError:
    backend = "nccl"
    print("Warning: SMDDP not found on this image, falling back to NCCL!")


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretraining Wav2Vec2 on Speech data")
 
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
        choices=["linear", "cosine", "cosine_with_restarts", 
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/opt/ml/checkpoints",
        help="Directory to save checkpoints during training.",

    )
    parser.add_argument(
        "--max_duration_in_seconds",
        type=float,
        default=5.0,
        help="Filter out audio files that are longer than `max_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--min_duration_in_seconds",
        type=float,
        default=3.0,
        help="Filter out audio files that are shorter than `min_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=None,
        help=(
            "If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the"
            " use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument(
        "--mask_time_prob",
        type=float,
        default=None,
        help=(
            "Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked in the"
            " contrastive task. If omitted, will pull value from model config."
        ),
    )
    parser.add_argument(
        "--mask_time_length",
        type=int,
        default=None,
        help=(
            "Length of each vector mask span to mask along the time axis in the contrastive task."
            " If omitted, will pull value from model config."
        ),
    )
    args = parser.parse_args()

    # Validate gradient accumulation steps
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")

    return args


def setup_distributed():
    try:
        if smddp.is_available:
            logger.info("Initializing SageMaker Distributed Data Parallel (SMDP).")
            dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = rank % torch.cuda.device_count()
            assert local_rank < torch.cuda.device_count(), f"Invalid local_rank {local_rank} exceeds available GPUs."

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


#
# Load Datasets
#

def save_checkpoint(model, optimizer, scheduler, epoch, args):
    checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, args):
    checkpoint_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pt')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getctime(os.path.join(args.checkpoint_dir, f)))
        checkpoint_path = os.path.join(args.checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch'] + 1
    return 0


def load_datasets_from_s3():
    """
    Lazily load and stream datasets from local SageMaker channel directory.
    """
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', '')

    if not train_dir:
        raise ValueError("Training data path (SM_CHANNEL_TRAIN) is not set.")

    def data_generator():
        """
        A generator that reads data line by line from all .txt files
        in the training directory.
        """
        for root, _, files in os.walk(train_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        for line in f:
                            try:
                                # Assume each line contains "<audio_path> <label>"
                                audio_file, label = line.strip().split(maxsplit=1)
                                absolute_audio_path = os.path.join(root, audio_file + ".flac")
                                yield {"audio": absolute_audio_path, "label": label}
                            except ValueError as e:
                                logger.warning(f"Skipping malformed line in {file_path}: {line}")
                                continue

    logger.info(f"Streaming dataset from directory: {train_dir}")
    return Dataset.from_generator(data_generator)



# def load_datasets_from_s3():
#     """
#     Loads dataset from local dataset directory that has been
#     setup by sagemaker
#     """
#     train_dir = os.environ.get('SM_CHANNEL_TRAIN', '')
#     #val_dir = os.environ.get('SM_CHANNEL_VALIDATION', '')

#     if not train_dir:
#         raise ValueError("Training or validation data path not provided")

#     raw_datasets = DatasetDict()
#     try:
#         logger.info(f"Loading training dataset from {train_dir}")
#         raw_datasets["train"] = load_dataset_audio(train_dir)

#         #logger.info(f"Loading validation dataset from {val_dir}")
#         #raw_datasets["validation"] = load_dataset_audio(val_dir)

#     except Exception as e:
#         logger.error(f"Error loading datasets from S3: {e}")
#         raise

#     logger.info(f"Training dataset size: {len(raw_datasets['train'])}")
#     #logger.info(f"Validation dataset size: {len(raw_datasets['validation'])}")

#     return raw_datasets


def rms_normalize_audio(audio_array, target_level=-25):
    """
    Root mean square audio normalization function
    """
    audio_array = audio_array.astype(np.float32)
    rms = np.sqrt(np.mean(audio_array**2))
    if rms > 0:
        target_rms = 10**(target_level/20)
        audio_array = audio_array * (target_rms/rms)
    return audio_array


def prepare_datasets(raw_datasets, feature_extractor, args):
    """
    Processes raw datasets using the given feature extractor.
    """
    # set max & min audio length in number of samples
    max_length = int(args.max_duration_in_seconds * feature_extractor.sampling_rate)
    min_length = int(args.min_duration_in_seconds * feature_extractor.sampling_rate)

    try:
        logger.info("Casting audio column to the correct sampling rate.")
        raw_datasets = raw_datasets.cast_column(
            args.audio_column_name,
            datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )
        if not feature_extractor.do_normalize:
            raise ValueError(
                "Training is only supported for normalized inputs. Make sure ``feature_extractor.do_normalize == True``"
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
                max_length=max_length,
                truncation=True
            )
            batch["input_values"] = inputs.input_values[0]
            batch["input_length"] = len(inputs.input_values[0])
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
        if min_length > 0.0:
            vectorized_datasets = vectorized_datasets.filter(
                lambda x: x > min_length,
                num_proc=args.preprocessing_num_workers,
                input_columns=["input_length"],
            )
        vectorized_datasets = vectorized_datasets.remove_columns("input_length")

        logger.info(f"Number of training examples: {len(vectorized_datasets['train'])}")
        #logger.info(f"Number of validation examples: {len(vectorized_datasets['validation'])}")
    except Exception as e:
        logger.error(f"Error mapping datasets: {e}")
        raise

    return vectorized_datasets


def setup_configs(args):
    try:
        logger.info(f"Loading model configuration from {args.model_name_or_path}")
        return Wav2Vec2Config.from_pretrained(args.model_name_or_path)
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        raise


def setup_model(args, config, device, local_rank):
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
            model = DDP(model, device_ids=[local_rank], 
                        output_device=local_rank,
                        find_unused_parameters=True)
            logger.info("Model wrapped in DistributedDataParallel")

        return model
    except Exception as e:
        logger.error(f"Error setting up model: {e}")
        raise


def setup_data_collator(args, config, model, feature_extractor):
    mask_time_prob = 0.65 if args.mask_time_prob is None else args.mask_time_prob
    mask_time_length = 10 if args.mask_time_length is None else args.mask_time_length

    logger.info(f"Using mask_time_prob={mask_time_prob}, mask_time_length={mask_time_length}")

    try:
        logger.info("Creating data collator for Wav2Vec2 pretraining.")
        data_collator = DataCollatorForWav2Vec2Pretraining(
            model=model,
            feature_extractor=feature_extractor,
            pad_to_multiple_of=args.pad_to_multiple_of,
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length,
        )
        logger.info("Data collator created successfully.")
    except Exception as e:
        logger.error(f"Error creating data collator: {e}")
        raise

    return data_collator


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
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

            device = batch["input_values"].device
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


def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)


def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


#
# Training Function
#
def main(args):
    if args.seed is not None:
        logger.info(f"Setting seed for reproducibility: {args.seed}")
        set_seed(args.seed)
    device, local_rank, world_size = setup_distributed()
    logger.info(f"Distributed setup complete. Device: {device}, Local rank: {local_rank}, World size: {world_size}")

    logger.info("Loading datasets from S3.")
    raw_dataset = DatasetDict({"train":load_datasets_from_s3()})

    try:
        logger.info(f"Loading feature extractor from {args.model_name_or_path}")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
    except Exception as e:
        logger.error(f"Error loading feature extractor: {e}")
        raise

    logger.info("Setting up model and data collator.")
    config = setup_configs(args)
    model = setup_model(args, config, device, local_rank)
    data_collator = setup_data_collator(args, config, model, feature_extractor)

    logger.info("Preparing datasets for training.")
    vectorized_datasets = prepare_datasets(raw_dataset, feature_extractor, args)
    # Check if lazy loading worked
    logger.info(f"Number of training examples: {len(vectorized_datasets['train'])}")
    #logger.info(f"Sample preprocessed data: {vectorized_datasets['train'][0]}")
    try:
        train_sampler = DistributedSampler(
            vectorized_datasets["train"],
            shuffle=True,
            num_replicas=dist.get_world_size() if is_distributed() else 1,
            rank=dist.get_rank() if is_distributed() else 0
        )

        logger.info(f"Rank {dist.get_rank()} has {len(train_sampler)} samples.")
        train_dataloader = DataLoader(
            vectorized_datasets["train"],
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            shuffle=True,
            # sampler=train_sampler,
            num_workers=args.preprocessing_num_workers,
            pin_memory=True,
            drop_last=True
        )
    except Exception as e:
        logger.error(f"Error creating DataLoaders: {e}")
        raise

    logger.info("Setting up optimizer and learning rate scheduler.")
    try:
        optimizer = AdamW(model.parameters(), lr=3e-4, betas=[args.adam_beta1, args.adam_beta2], eps=args.adam_epsilon)
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
    try:    
    # Load checkpoint if exists
        start_epoch = load_checkpoint(model, optimizer, lr_scheduler, args)
    except:
        start_epoch = 0    
    scaler = GradScaler()
    completed_steps = 0
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        total_network_latency = 0
        num_network_operations = 0
        num_steps = 0
    
        logger.info(f"Starting training for epoch {epoch}.")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        iterator = train_dataloader
        if not is_distributed() or dist.get_rank() == 0:
            iterator = tqdm(iterator, desc=f"Training Epoch {epoch}")
            logger.info("Iterator successfully created")
    
        model.train()
        for step, batch in enumerate(iterator):
            try:
                # Move batch to device
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                network_start_time = time.time()
    
                # Forward pass
                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss / args.gradient_accumulation_steps
                    logger.info(f"Loss: {loss}")
    
                # Backward pass with scaled loss
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
    
                # Gradient scaling based on num_losses
                num_losses = batch["mask_time_indices"].sum()
                if is_distributed():
                    # Synchronize num_losses across all processes
                    num_losses_tensor = num_losses.clone()
                    dist.all_reduce(num_losses_tensor, op=dist.ReduceOp.SUM)
                    num_losses = num_losses_tensor.item()
                    gradient_multiplier = dist.get_world_size() / num_losses
                    multiply_grads(model.module.parameters(), gradient_multiplier)
                else:
                    multiply_grads(model.parameters(), 1 / num_losses.item())

                # Gradient accumulation and optimizer step
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # Compute gradient norm for monitoring
                    scale = scaler.get_scale() if hasattr(scaler, "get_scale") else 1
                    grad_norm = get_grad_norm(
                        model.module.parameters() if hasattr(model, "module") else model.parameters(),
                        scale=scale,
                    )
                    logger.info(f"Gradient norm: {grad_norm:.4f}")
    
                    # Check for NaN/Inf in gradients
                    # nan_count = 0
                    # inf_count = 0
                    # temp_model_ref = model.module if hasattr(model, "module") else model
                    # for param in temp_model_ref.parameters():
                    #     if param.grad is not None:
                    #         if torch.isnan(param.grad).any():
                    #             nan_count += 1
                    #         if torch.isinf(param.grad).any():
                    #             inf_count += 1
    
                    # if nan_count > 0 or inf_count > 0:
                    #     logger.warning(f"NaN count: {nan_count}, Inf count: {inf_count}, Step: {step}")
    
                    # Perform optimizer step and zero_grad
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
    
                    # Update metrics
                    completed_steps += 1
                    num_steps += 1
                    network_latency = time.time() - network_start_time
                    total_network_latency += network_latency
                    num_network_operations += 1
    
                # Log all results at defined intervals
                # if (step + 1) % (args.gradient_accumulation_steps) == 0:
                #     loss_val = loss.detach()
                #     contrastive_loss = outputs.contrastive_loss.detach()
                #     diversity_loss = outputs.diversity_loss.detach()
    
                #     if is_distributed():
                #         loss_tensor = loss_val.clone()
                #         dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                #         loss_val = loss_tensor.item()
                #         contrastive_loss_tensor = contrastive_loss.clone()
                #         dist.all_reduce(contrastive_loss_tensor, op=dist.ReduceOp.SUM)
                #         contrastive_loss = contrastive_loss_tensor.item()
                #         diversity_loss_tensor = diversity_loss.clone()
                #         dist.all_reduce(diversity_loss_tensor, op=dist.ReduceOp.SUM)
                #         diversity_loss = diversity_loss_tensor.item()
                #         percent_masked_tensor = batch["mask_time_indices"].float().sum().clone()
                #         dist.all_reduce(percent_masked_tensor, op=dist.ReduceOp.SUM)
                #         percent_masked = percent_masked_tensor.item()
                #     else:
                #         percent_masked = batch["mask_time_indices"].float().sum().item()
    
                #     train_logs = {
                #         "loss": torch.tensor((loss_val * args.gradient_accumulation_steps) / num_losses),
                #         "contrast_loss": torch.tensor(contrastive_loss / num_losses),
                #         "div_loss": torch.tensor(diversity_loss / num_losses),
                #         "%_mask_idx": torch.tensor(percent_masked / (dist.get_world_size() if is_distributed() else 1)),
                #         "ppl": outputs.codevector_perplexity,
                #         "lr": torch.tensor(optimizer.param_groups[0]["lr"]),
                #         "grad_norm": torch.tensor(grad_norm),
                #     }
                    
                #     log_str = "".join(f"| {k}: {v.item():.3e}" for k, v in train_logs.items())
                #     logger.info(log_str)

    
            except Exception as e:
                logger.error(f"Error during training step {step} of epoch {epoch}: {e}")
                if is_distributed():
                    cleanup_distributed()
                raise
    
        # Epoch-level metrics
        epoch_duration = time.time() - epoch_start_time
        throughput = (len(train_dataloader) * args.per_device_train_batch_size * dist.get_world_size()) / epoch_duration
        avg_network_latency = total_network_latency / num_network_operations if num_network_operations > 0 else 0
    
        logger.info(f"Epoch {epoch + 1} training_time: {epoch_duration:.2f} seconds")
        logger.info(f"Epoch {epoch + 1} throughput: {throughput:.2f} samples/second")
        logger.info(f"Epoch {epoch + 1} avg_network_latency: {avg_network_latency:.2f} seconds")

        # Save checkpoint at the end of each epoch
        save_checkpoint(model, optimizer, lr_scheduler, epoch, args)
    
    cleanup_distributed()



if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting main training script.")
    try:
        main(args)
    except Exception as e:
        logger.error(f"Fatal error in main script: {e}")
        raise
