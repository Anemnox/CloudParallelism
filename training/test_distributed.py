import os
import concurrent.futures
import time
import threading

from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker import Session
import boto3

# Initialize session and other common variables
session = boto3.Session()
role = os.getenv("SAGEMAKER_ROLE")  # Role ARN that has permission to execute training jobs in SageMaker
region = session.region_name
bucket = "tcss562-west"  # Replace with your S3 bucket name

# Set up S3 paths for input data and model output
train_data_s3_uri = f"s3://{bucket}/small-test"
validation_data_s3_uri = f"s3://{bucket}/small-test/"
output_path = f"s3://{bucket}/training-output-test/"

# Define different configurations to test
training_script = "training.py"
QUOTA = 8
SEED_NUMBER = 42
instance_type = "ml.g4dn.2xlarge"
configurations = [
    {"instance_count": 2, "epochs": 10}
]

current_instance_usage = 0
lock = threading.Lock()


def can_launch_test(resource_request):
    return current_instance_usage + resource_request <= QUOTA


# Function to create and launch a training job with a given configuration
def train_with_config(config):
    global current_instance_usage
    
    estimator = PyTorch(
        entry_point=training_script,
        source_dir=".",
        role=role,
        framework_version="1.11.0",
        py_version="py38",
        hyperparameters={
            "model_name_or_path": "facebook/wav2vec2-base-960h",
            "gradient_accumulation_steps": (
                config["gradient_accumulation_steps"] if
                "gradient_accumulation_steps" in config else 4),
            "per_device_train_batch_size": (
                config["per_device_train_batch_size"] if
                "per_device_train_batch_size" in config else 4),
            "epochs": config["epochs"] if "epochs" in config else 1,
            "lr_scheduler_type": "linear",
            "max_duration_in_seconds": (
                config["max_duration_in_seconds"] if
                "max_duration_in_seconds" in config else 5.0),
            "seed": SEED_NUMBER,
            "preprocessing_num_workers": 1,
            "backend": "smddp",
        },
        environment={
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_IB_DISABLE": "0",
            "NCCL_P2P_DISABLE": "0",
        },
        output_path=output_path,
        sagemaker_session=Session(boto_session=session),
        instance_count=config["instance_count"],
        instance_type=instance_type,
    )

    train_input = TrainingInput(train_data_s3_uri)
    validation_input = TrainingInput(validation_data_s3_uri)

    print(f"Starting training with config: {config}")

    # Start Metric
    
    estimator.fit({
        "train": train_input,
        "validation": validation_input
    })

    # Stop Metric

    current_instance_usage -= config["instance_count"]
    return f"Training completed for config: {config}"


# Execute the training jobs in parallel
def main():
    global current_instance_usage
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for config in configurations:
            resource_need = config["instance_count"]
            
            # Wait until condition is met
            while not can_launch_test(resource_need):
                time.sleep(10)  # Delay to avoid busy-waiting
            
            with lock:
                current_instance_usage += resource_need
                futures.append(executor.submit(train_with_config, config))
        
        for future in concurrent.futures.as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()