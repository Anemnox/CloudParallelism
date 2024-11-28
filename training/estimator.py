import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import boto3
import os

# First configure AWS Credentials?
# - aws configure (to set up)
# - may need to setup SAGEMAKER_ROLE


# Initialize boto3 session for SageMaker
session = boto3.Session()
role = os.getenv("SAGEMAKER_ROLE")  # Role ARN that has permission to execute training jobs in SageMaker
region = session.region_name
bucket = "tcss562"  # Replace with your S3 bucket name

# Set up S3 paths for input data and model output
train_data_s3_uri = f"s3://{bucket}/LibriSpeech/dev-clean"
validation_data_s3_uri = f"s3://{bucket}/LibriSpeech/dev-clean/"
output_path = f"s3://{bucket}/training-output-test/"


# Define SageMaker PyTorch Estimator
estimator = PyTorch(
    entry_point="training.py",  # Your training script
    source_dir=".",  # Directory containing the script and dependencies
    role=role,
    framework_version="1.11.0",
    py_version="py38",
    hyperparameters={
        "model_name_or_path": "facebook/wav2vec2-base-960h",  # Fixed the model path parameter
        "gradient_accumulation_steps": 4,
        "per_device_train_batch_size": 8,
        "epochs": 1,
        "lr_scheduler_type": "linear",
        "max_duration_in_seconds": 5.0,
        "seed": 42,  # Add seed for reproducibility
        "preprocessing_num_workers": 1,
        "backend": "smddp",
        # "mpi": {
        #     "enabled": True,
        #     "processes_per_host": 8,  # Number of GPUs per instance
        #     "custom_mpi_options": "-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH"
        # },
        #"checkpoint_dir": f"s3://{bucket}/checkpoint",
    },
    environment={
        "NCCL_DEBUG": "INFO",
        "NCCL_SOCKET_IFNAME": "eth0",  # Ensure network interface compatibility
        "NCCL_IB_DISABLE": "0",       # Enable InfiniBand if available
        "NCCL_P2P_DISABLE": "0",      # Enable peer-to-peer communication
    },
    output_path=output_path,
    sagemaker_session=sagemaker.Session(boto_session=session),

    instance_count=1,  # Number of instances for distributed training
    instance_type="ml.c5.xlarge" # "ml.p3.2xlarge",  # Instance type, can be adjusted based on needs
    #use_spot_instances=True,  # Enable spot instances
)

# Specify the training data inputs
train_input = TrainingInput(train_data_s3_uri)
validation_input = TrainingInput(validation_data_s3_uri)

# Launch the training job
estimator.fit({
    "train": train_input,
    "validation": validation_input
})

print("Training job started!")