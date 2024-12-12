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
role = "<insert role>"
region = session.region_name
bucket = "tcss562-cloud-computing-project"

# Set up S3 paths for input data and model output
train_data_s3_uri = f"s3://{bucket}/small-test/train-clean-100/"
#train_data_s3_uri = f"s3://{bucket}/small-test/dev-clean/"
validation_data_s3_uri = f"s3://{bucket}/small-test/"
output_path = f"s3://{bucket}/training-output/"

# Define different configurations to test
training_script = "training.py"
QUOTA = 16
SEED_NUMBER = 42
instance_type = "ml.g4dn.12xlarge"
configurations = [
    #{"instance_count": 1, "epochs": 1},
    {"instance_count": 2, "epochs": 1},
    #{"instance_count": 3, "epochs": 1},
    #{"instance_count": 4, "epochs": 1},
    #{"instance_count": 8, "epochs": 1},
    #{"instance_count": 12, "epochs": 1}  # Uncomment if quota allows
    #{"instance_count": 16, "epochs": 1}
]

# Define metric definitions
metric_definitions = [
    {'Name': 'network_latency', 'Regex': 'Epoch \\d+ avg_network_latency: ([0-9\\.]+) seconds'},
    {'Name': 'epoch_training_time', 'Regex': 'Epoch \\d+ training_time: ([0-9\\.]+) seconds'},
    {'Name': 'epoch_throughput', 'Regex': 'Epoch \\d+ throughput: ([0-9\\.]+) samples/second'},
    {'Name': 'loss', 'Regex': 'Loss: ([0-9\\.]+)'},
    {'Name': 'gradient_norm:', 'Regex': 'Gradient norm: ([0-9\\.]+)'},
]

current_instance_usage = 0
lock = threading.Lock()

def can_launch_test(resource_request):
    return current_instance_usage + resource_request <= QUOTA

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
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 4),
            "per_device_train_batch_size": config.get("per_device_train_batch_size", 4),
            "epochs": config.get("epochs", 1),
            "lr_scheduler_type": "linear",
            "seed": SEED_NUMBER,
            "preprocessing_num_workers": 4,
        },
        distribution={
            "smdistributed": {
                "dataparallel": {
                    "enabled": True,
                    "custom_mpi_options": "-verbose -x NCCL_DEBUG=VERSION"
                }
            }
        },
        environment={
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "ALL",
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_IB_DISABLE": "1",  
            "NCCL_P2P_DISABLE": "1",  
            "NCCL_SHM_DISABLE": "0"   
        },
        output_path=output_path,
        sagemaker_session=Session(boto_session=session),
        instance_count=config["instance_count"],
        instance_type=instance_type,
        use_spot_instances=True, # Enable spot instances
        max_run=3600*3,
        max_wait=3600*3,  # Max time including interruptions
        max_retry_attempts=30,
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/",  # Save checkpoints to S3
        checkpoint_local_path='/opt/ml/checkpoints',
        enable_sagemaker_metrics=True,
        metric_definitions=metric_definitions
    )

    train_input = TrainingInput(train_data_s3_uri)
    validation_input = TrainingInput(validation_data_s3_uri)

    print(f"Starting training with config: {config}")

    estimator.fit({
        "train": train_input,
        "validation": validation_input
    })

    with lock:
        current_instance_usage -= config["instance_count"]
    return f"Training completed for config: {config}"

def main():
    global current_instance_usage
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for config in configurations:
            resource_need = config["instance_count"]

            while not can_launch_test(resource_need):
                time.sleep(10)

            with lock:
                current_instance_usage += resource_need
                futures.append(executor.submit(train_with_config, config))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()