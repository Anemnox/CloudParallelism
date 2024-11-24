import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import boto3
import os

# Initialize boto3 session for SageMaker
session = boto3.Session()
role = os.getenv("SAGEMAKER_ROLE")  # Role ARN that has permission to execute training jobs in SageMaker
region = session.region_name
bucket = "tcss562-asr-data"  # Replace with your S3 bucket name

# Set up S3 paths for input data and model output
train_data_s3_uri = f"s3://{bucket}/training/"
validation_data_s3_uri = f"s3://{bucket}/validation/"
output_path = f"s3://{bucket}/training-output/"

# Define SageMaker PyTorch Estimator
estimator = PyTorch(
    entry_point="training.py",  # Your training script
    source_dir=".",  # Directory containing the script and dependencies
    role=role,
    framework_version="1.9.0",
    py_version="py38",
    instance_count=2,  # Number of instances for distributed training
    instance_type="ml.p3.2xlarge",  # Instance type, can be adjusted based on needs
    hyperparameters={
        "model_name_or_path": "facebook/wav2vec2-base-960h",  # Fixed the model path parameter
        "gradient_accumulation_steps": 4,
        "per_device_train_batch_size": 8,
        "epochs": 1,
        "lr_scheduler_type": "linear",
        "max_duration_in_seconds": 5.0,  
        "seed": 42  # Add seed for reproducibility
    },,
    output_path=output_path,
    sagemaker_session=sagemaker.Session(boto_session=session)
)

# Specify the training data inputs
train_input = TrainingInput(train_data_s3_uri, content_type="text/csv")
validation_input = TrainingInput(validation_data_s3_uri, content_type="text/csv")

# Launch the training job
estimator.fit({
    "train": train_input,
    "validation": validation_input
})

print("Training job started!")
