# Build On Trainium Resources
**Purpose:** 

Collection of resources (documentation, examples, tutorials and workshops) to help onboard new students and researchers. This set of resources will need to updated and maintained as new resources become available. 

# Resources

This section contains links to various documentation sources and is a helpful index when working on Neuron. It is organized into several sections based on workload and relevance.

## Getting Started with Neuron

|Title	|Description	|Link	|
|---	|---	|---	|
|Getting Started with AWS	|Getting started resource for AWS, generally, including AWS environment provisioning, budger alarms, CLI, instance setip and best practices for working in an AWS environment	|[BoT Getting Started on AWS](https://github.com/scttfrdmn/aws-101-for-tranium)	|
|Neuron Documentation	|The Neuron Official Product Documentation. This contains details on our software libraries and hardware.	|[Neuron Docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html)	|
|Inf2 Instance Details	|Helpful overview links for the Inferentia2 Instance and associated accelerators	|<ul><li>[AWS Landing Page](https://aws.amazon.com/ai/machine-learning/inferentia/) </li><li> [Instance Details](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inf2-arch.html#aws-inf2-arch) </li><li> [Chip Details](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inferentia2.html#inferentia2-arch) </li><li> [Core Details](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v2.html#neuroncores-v2-arch) </li></ul> |
|Trn1 Instance Details	|Similar overview links for Trn1 instances and acclerators	|<ul><li>[AWS Landing Page](https://aws.amazon.com/ai/machine-learning/trainium/) </li><li>[Instance Details](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trn1-arch.html#aws-trn1-arch) </li><li> [Chip Details](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium.html#trainium-arch) </li><li> [Core Details](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v2.html#neuroncores-v2-arch) </li></ul>	|
|Trn2 Instance Details	|Similar overview links for Trn2 instances and acclerators	|<ul><li>[Youtube Launch Video](https://www.youtube.com/watch?v=Bteba8KLeGc) </li><li> [Instance Details](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trn2-arch.html#aws-trn2-arch) </li><li> [Chip Details](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium2.html#trainium2-arch) </li><li> [Core Details](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v3.html#neuroncores-v3-arch) </li></ul>	|
| Instance Service Quotas  | Understand what service quotas are, how they apply to Inferentia and Trainium instances and endpoints, and have an example of what quotas would be appropriate for a POC. |[Inferentia and Trainium Service Quotas](https://repost.aws/articles/ARgmEMvbR6Re200FQs8rTduA/inferentia-and-trainium-service-quotas) |
|Software Overview - General	|Overview Video of Trainium Software Stack	|[Video](https://www.youtube.com/watch?v=vaqj8XQfqwM&t=806s)	|
|Software Overview - Framework	|Application Frameworks for developing on Neuron. Torch-NeuronX for small model inference and training, NxD for Distributed modeling primitives, NxDI - a higher abstraction library for inference and NxDT a corresponding abstraction for training.	|<ul><li>Torch-NeuronX ([Training](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/programming-guide/training/pytorch-neuron-programming-guide.html#pytorch-neuronx-programming-guide), [Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/programming-guide/inference/trace-vs-xla-lazytensor.html)) </li><li> [NxD](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/developer-guide.html) </li><li> [NxD-T](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/overview.html#nxd-training-overview) </li><li> [NxD-I](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/nxdi-overview.html#nxdi-overview) </li></ul>	|
|Software Overview - ML Libraries	|ML libraries which offer another interface for deploying to trn/inf. Optimum-Neuron provides and interface between transformers and AWS Accelerators. AXLearn is a training library built on top of JAX and XLA.	|[Optimum Neuron](https://huggingface.co/docs/optimum-neuron/index) [AXLearn](https://github.com/apple/axlearn)	|
|Environment Setup	|A set of resources on provisioning instances and setting up development environments with the appropriate Neuron Software.	|<ul><li>[Instance Guide](https://repost.aws/articles/ARTxLi0wndTwquyl7frQYuKg) </li><li> [Remote Development Guide](https://repost.aws/articles/ARmgDHboGkRKmaEyfBzyVP4w) </li><li> [AMIs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/dlami/index.html)</li><li> [Containers](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/containers/index.html) </li><li> [Manual Setup](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html#setup-torch-neuronx-ubuntu22)	</li></ul> |
|Release Versions	|Index of the latest release versions and their semantic version information.	|<ul><li>[Latest Release Version](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html#latest-neuron-release)</li><li>[Component Package Verisons](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/releasecontent.html#latest-neuron-release-artifacts)</li><li>[Deploying previous SDK versions](https://builder.aws.com/content/30beZEzf3XencHUcq7XDVLClkQ6/deploying-previous-versions-of-neuron-sdk-on-trainium-and-inferentia-ec2-instances)</li></ul>|

## Training Resources

|Title	|Description	|Link	|
|---	|---	|---	|
|Torch-NeuronX Docs	|Torch-NeuronX docs on the XLA flow, and constructing a simple training loop on Trainium/Inferentia.	|[Torch-NeuronX Training Docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/programming-guide/training/pytorch-neuron-programming-guide.html#pytorch-neuronx-programming-guide)	|
|NxD Docs	|Details on NxD, as well as the Distributed Layer Primitives (Tensor Parallelism, Pipeline Parallelism, etc.)	|[NxD Developer Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/developer-guide-training.html#neuronx-distributed-developer-guide-training)	|
|NxD Docs + PyTorch Lightning	|PyTorch Lightning Docs for NxD Training	|[PTL Developer Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/ptl_developer_guide.html#ptl-developer-guide)	|
|NxD-T Developer Guide	|NxD-Training, A higher level abstraction library on NxD for training specific workloads.	|[NxD Training Developer Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/overview.html#nxd-training-overview)	|
|PreTraining	|Pre-Training samples within various different libraries above	|<ul><li>[Torch-NeuronX](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/mlp.html#neuronx-mlp-training-tutorial)</li><li>[NXD](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tutorials/training_llama_tp_zero1.html#llama2-7b-tp-zero1-tutorial)</li><li>[NxD-T](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/tutorials/hf_llama3_8B_pretraining.html#hf-llama3-8b-pretraining)</li><li>[Optimum Neuron](https://huggingface.co/docs/optimum-neuron/training_tutorials/pretraining_hyperpod_llm)</li></ul> |
|LoRA Fine Tuning	|LoRA Samples within the various libraries for Neuron	|<ul><li>[NxD-T](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/tutorials/hf_llama3_8B_SFT_LORA.html#hf-llama3-8b-sft-lora)</li><li>[Optimum Neuron](https://huggingface.co/docs/optimum-neuron/training_tutorials/sft_lora_finetune_llm) </li></ul>	|
|Preference Alignment	|Preference Alignment Samples within the various libraries for Neuron	|[NxD-T](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/tutorials/hf_llama3_8B_DPO_ORPO.html#hf-llama3-8b-dpo-orpo)	|
|Awsome Distributed Training	|Reference Distributed Training Examples on AWS	|[Awsome-distributed-training](https://github.com/aws-samples/awsome-distributed-training)	|

## Inference Resources

|Title	|Description	|Link	|
|---	|---	|---	|
|Torch-NeuronX Docs	|Torch-NeuronX docs on the XLA flow, and tracing models for Inference on a single core. Samples of various common models as well.	|[Torch-NeuronX Docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/inference/api-torch-neuronx-trace.html#torch-neuronx-trace-api)
[Samples](https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference)	|
|NxD-I Developer Guide	|NxD-Inference, A higher level abstraction library on NxD for inference specific workloads.	|[NxD-I Docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/index.html)	|
|Deployment vLLM	|Guide for vLLM development with NxDI	|[vLLM Docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide.html)	|
|TGI	|Guide on how to use HuggingFace Text Generation Inference (TGI) with Neuron	|[TGI Docs](https://huggingface.co/docs/optimum-neuron/en/guides/neuronx_tgi)	|

## Kernel Resources

|Title	|Description	|Link	|
|---	|---	|---	|
|NKI Docs	|General NKI docs 	|[NKI Docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)	|
|Getting Started With NKI	|Getting started writing NKI Kernels 	|[Getting Started With NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/getting_started.html#nki-getting-started)	|
|Performant Kernels with NKI	|Understanding NKI kernel performance	|[Performant Kernels with NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/nki_arch_guides.html#nki-arch-guides)	|
|NKI - Sample Kernels	|Sample Kernel Repository with reference implementation	|[NKI - Sample Kernels](https://github.com/aws-neuron/nki-samples/tree/main)	|

## Tools Resources

|Title	|Description	|Link	|
|---	|---	|---	|
|Profiler	|Neuron Profiler User Guide 	|[Profiler Docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profiler-2-0-beta-user-guide.html)	|
|Monitoring Tools and CLI	|Monitoring and CLI tools for working with Neuron Hardware.	|[Monitoring Tools and CLI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html)	|

# Learning Paths

Learning Paths are a list or organized exercises 

## Training 

|Title	|Description	|Link	|Minimum Instance Required	|
|---	|---	|---	|---	|
|Setup an Instance/Developer Environment	|This section contains resources to provision a developer Environment. This is a great starting place if you need a clean environment for development, or for starting any of the following exercises.	|<ul><li> [Instance Setup](https://repost.aws/articles/ARTxLi0wndTwquyl7frQYuKg) <li></li> [DLAMIs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/dlami/index.html) </li></ul> 	|trn1.2xlarge	|
|Construct a simple Training Loop with torch-neuronx	|This is a sample of how to construct a training loop using torch-neuronx. Relevant for getting started with XLA flows, as well as models which require a single core/DP.	|[MLP Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/mlp.html#neuronx-mlp-training-tutorial)	|trn1.2xlarge	|
|Implement Tensor Parallelism with NeuronX Distributed	|Implement Tensor Parallel for a model to shard training across accelerators.	|[BERT Pretraining Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tutorials/training.html#tp-training-tutorial)	|trn1.32xlarge	|
|Pre-training Llama with TP, PP and ZeRO-1	|Train a model using multiple forms of parallelism (Tensor Parallelism, Pipeline Parallelism, and ZeRO-1). This uses the NxD Core Library and should give a good view of the parallel primatives.	|[Llama Pretraining Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tutorials/training_llama_tp_zero1.html)	|4x trn1.32xlarge cluster	|
|LoRA Fine Tuning with Optimum Neuron	|Fine-Tune a model with LoRA on Optimum Neuron. Optimum Neuron is a library developed by HF and allows for simple modifications to transformers code to port to Neuron.	|[Qwen LoRA Optimum Neuron](https://huggingface.co/docs/optimum-neuron/training_tutorials/qwen3-fine-tuning)	|trn1.32xlarge	|
|LoRA Fine-Tuning with NxDT	|LoRA based Fine-tune a model using NxD-T, our higher level training library built on top of NxD core.	|[LoRA NxDT Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/tutorials/hf_llama3_8B_SFT_LORA.html#hf-llama3-8b-sft-lora)	|trn1.32xlarge	|
|DPO/ORPO Fine-Tuning with NxDT	|Preference Alignment for a model using NxD-T, our higher level training library built on top of NxD core.	|[DPO/ORPO Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/tutorials/hf_llama3_8B_DPO_ORPO.html)	|trn1.32xlarge	|



## Inference Path

|Title	|Description	|Link	|Minimum Instance Required	|
|---	|---	|---	|---	|
|Setup an Instance/Developer Environment	|This section contains resources to provision a developer Environment. This is a great starting place if you need a clean environment for development, or for starting any of the following exercises.	|<ul><li> [Instance Setup](https://repost.aws/articles/ARTxLi0wndTwquyl7frQYuKg) <li></li> [DLAMIs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/dlami/index.html) </li></ul> 	|trn1.2xlarge	|
|Trace Models with Torch-NeuronX	|Trace small models without model parallelism for inference with torch-neuronx.	|[Torch-NeuronX Tutorials](https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/README.md#inference)	|trn1.2xlarge	|
|Deploy Various Models with Optimum Neuron	|Optimum Neuron allows for popular models in diffusers and transformers to easily be deployed to Neuron devices.	|[Optimum Neuron Tutorials](https://huggingface.co/docs/optimum-neuron/inference_tutorials/notebooks)	|trn1.32xlarge	|
|Deploy LLM with NxD	|NxD is our library with model sharding primitives. This guide serves as a good jumping off point for common LLMs	|[NxD-I Production Models](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/model-reference.html)	|trn1.32xlarge	|
|vLLM Integration	|This guide walks through how to run models with vLLM on Neuron devices. This uses the previously mentioned NxDI back-end for the model deployments.	|[vLLM User Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide.html)	|trn1.32xlarge	|
|Deploy a DiT with NxD	|This guide walks through a non LLM model architecture to be sharded and deployed on Neuron. In this case it is a Diffusion Transformer architecture for image generation	|[PixArt Sigma on Neuron](https://aws.amazon.com/blogs/machine-learning/cost-effective-ai-image-generation-with-pixart-sigma-inference-on-aws-trainium-and-aws-inferentia/)	|trn1.32xlarge	|
|Onboard a new Model to NxD-I	|This guide walks through how to onboard a new model to NxD	|[Model Onboarding Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/onboarding-models.html)	|trn1.32xlarge	|
|Explore Additonal features of NxD-O	|Here are a few additonal references for NxD-I feature that may be rel;evant for your specific use case (Multi-LoRA, Quantization, Spec. decode)	|<ul><li>[Quantization](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/custom-quantization.html)<li></li> [Spec. Decode](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/trn2-llama3.3-70b-tutorial.html#nxdi-trn2-llama3-3-70b-tutorial)<li></li> [Multi-LoRA](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/trn2-llama3.1-8b-multi-lora-tutorial.html#nxdi-trn2-llama3-1-8b-multi-lora-tutorial) </li></ul> 	|trn1.32xlarge	|

## Kernel/Compiler Path

|Title	|Description	|Link	|Minimum Instance Required	|
|---	|---	|---	|---	|
|Setup an Instance/Developer Environment	|This section contains resources to provision a developer Environment. This is a great starting place if you need a clean environment for development, or for starting any of the following exercises.	|<ul><li> [Instance Setup](https://repost.aws/articles/ARTxLi0wndTwquyl7frQYuKg) <li></li> [DLAMIs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/dlami/index.html) </li></ul> 	|trn1.2xlarge	|
|Writing Functional Kernels	|This Getting Started Guide will demonstrate how to write a Hello World, element-wise tensor add kernel. This will give you a good foundation for reading and understanding the other kernels.	|[Getting Started with NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/getting_started.html#nki-getting-started)	|trn1.2xlarge	|
|NKI workshop	|This workshop walks through how to build, profile and integrate a kernel into PyTorch modelling.	|[NKI Workshop](https://github.com/aws-samples/ml-specialized-hardware/tree/main/workshops/03_NKIWorkshop)	|trn1.2xlarge	|
|Walkthrough NKI Tutorials	|These tutorials walkthrough popular kernels and the associated optimizations applied. This is a good set of kernels to show how to iteratively write and optimize kernels.	|[NKI Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials.html)	|trn1.2xlarge	|
|Review NKI Samples	|This repository contains the implementations of optimized reference kernels, used within our serving libraries and implementations.	|[NKI Samples](https://github.com/aws-neuron/nki-samples/)	|trn1.2xlarge	|
|Profiling NKI Kernels	|This guide walks through how to profile kernels and use the Neuron Profiler	|[Profiling NKI Kernels](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/neuron_profile_for_nki.html#neuron-profile-for-nki)	|trn1.2xlarge	|

# Appendix

## Other Resources

|Title	|Description	|Link	|
|---	|---	|---	|
|Re:Invent 2024 Recap	|REcap Post from Re:Invent, which includes links to workshops and sessions on Neuron	|[RePost Article](https://repost.aws/articles/ARuhbPQliOSqKn74zJpGmMYQ)	|
|AI on EKS	|Reference implementation for AI workloads on EKS including hosting on Trainium	|[AI on EKS](https://github.com/awslabs/ai-on-eks)	|
