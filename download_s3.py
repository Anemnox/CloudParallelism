import os
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# AWS S3 bucket details
BUCKET_NAME = "tcss562-cloud-computing-project"  # Replace with your bucket name
LOCAL_DIRECTORY = "LibriSpeech_test"  # Replace with your directory path
S3_DIRECTORY = "small-test"  # Replace with your desired S3 folder path (can be empty)

# Initialize S3 client
s3_client = boto3.client('s3')

def upload_file(file_path, s3_key):
    """
    Uploads a file to S3.
    """
    try:
        s3_client.upload_file(file_path, BUCKET_NAME, s3_key)
        return f"Success: {file_path} -> {s3_key}"
    except Exception as e:
        return f"Error uploading {file_path}: {e}"

def upload_files_parallel(local_dir, s3_dir, max_workers=10):
    """
    Uploads files from a local directory to S3 in parallel with a progress bar.
    """
    # Collect all files in the directory
    files_to_upload = []
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Create S3 key based on relative path
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(s3_dir, relative_path).replace("\\", "/")  # For Windows compatibility
            files_to_upload.append((local_path, s3_key))

    
    print(f"files to upload: {len(files_to_upload)}")
    # Upload files using ThreadPoolExecutor with a progress bar
    results = []
    with tqdm(total=len(files_to_upload), desc="Uploading Files", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(upload_file, file_path, s3_key): file_path for file_path, s3_key in files_to_upload}
            for future in as_completed(future_to_file):
                results.append(future.result())
                pbar.update(1)  # Increment progress bar
    
    return results

if __name__ == "__main__":
    max_workers = 20  # Adjust the number of threads
    results = upload_files_parallel(LOCAL_DIRECTORY, S3_DIRECTORY, max_workers=max_workers)
    
    print("Done?!")
