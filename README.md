# Cloud Parallelism Project
Cloud Parallelism Project for TCSS 562 Fall 2024

[Link to doc](https://docs.google.com/document/d/19w5DobRjBiq_SC0tPWki5kRT00WHaMT8jWD0k50QCus/edit?usp=sharing)

SUMMARY:
1. Designed and implemented a distributed training pipeline for Wav2Vec2 on the 100-hour LibriSpeech dataset, leveraging AWS SageMaker's smdistributed module for multi-GPU parallelism.

2. Conducted comparative analysis of single-GPU (ml.g4dn.2xlarge) vs. multi-GPU (ml.g4dn.12xlarge) setups, isolating the impact of hardware distribution on training throughput, GPU utilization, and network latency.

3. Experimentally determined that multi-GPU setups reduced epoch training time by up to 50% and nearly doubled throughput (e.g., from 30.8 to 61.3 samples/second for 4 GPUs), while lowering inter-node communication latency by 7.6%.

4. Analyzed cost-performance tradeoffs between on-demand and spot instance clusters, showing that spot instances achieved up to 35% cost savings with minimal performance trade-offs, reducing total training costs by $7.96 for 12 GPUs.

5. Identified diminishing returns in scaling GPU clusters beyond 8 GPUs due to data pipeline and synchronization bottlenecks, highlighting critical areas for optimizing distributed training.
Objective: To evaluate and optimize GPU hardware and pricing configurations to achieve efficient, cost-effective training of large-scale Automatic Speech Recognition models.

References

1. [Distributed Training Concepts](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html#distributed-training-basic-concepts)

2. [Getting Started with Distributed Training on Sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training-get-started.html)

3. [Workshop on Distributed Training with Sagemaker](https://github.com/aws-samples/sagemaker-distributed-training-workshop)

4. [Pretraining Wav2Vec2 on Librispeech data](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-pretraining)

5. [Finetuning Wav2Vec2 on Sagemaker](https://github.com/aws-samples/amazon-sagemaker-fine-tune-and-deploy-wav2vec2-huggingface/blob/main/sagemaker-fine-tune-wav2vec2.ipynb)

