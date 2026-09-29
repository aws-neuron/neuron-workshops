# Neuron Workshops

In this workshop you will learn how to develop support for a new model with [NeuronX Distributed Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/index.html), through the context of Llama 3.2 1B. You will also learn how to write your own kernel to directly program the accelerated hardware with the [Neuron Kernel Interface](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html). These tools will help you design your research proposals and experiments on Trainium.

It also includes an end-to-end example of using Hugging Face Optimum Neuron to fine-tune and host a small language model with Amazon SageMaker.


### What are AWS Trainium and Neuron?
AWS Trainium is an AI chip developed by AWS for accelerating building and deploying machine learning models. Built on a specialized architecture designed for deep learning, Trainium accelerates the training and inference of complex models with high output and scalability, making it ideal for academic researchers looking to optimize performance and costs. This architecture also emphasizes sustainability through energy-efficient design, reducing environmental impact. Amazon has established a dedicated Trainium research cluster featuring up to 40,000 Trainium chips, accessible via Amazon EC2 Trn1 instances. These instances are connected through a non-blocking, petabit-scale network using Amazon EC2 UltraClusters, enabling seamless high-performance ML training. The Trn1 instance family is optimized to deliver substantial compute power for cutting-edge AI research and development. This unique offering not only enhances the efficiency and affordability of model training but also presents academic researchers with opportunities to publish new papers on underrepresented compute architectures, thus advancing the field.

Learn more about Trainium [here](https://aws.amazon.com/ai/machine-learning/trainium/).

### Your workshop
This hands-on workshop is designed for developers, data scientists, and machine learning engineers who are getting started in their journey on the Neuron SDK. 

The workshop has multiple available modules:
1. Set up instructions
2. Run inference with Llama and NeuronX Distributed inference (NxD)
3. Write your own kernel with Neuron Kernel Interface (NKI)
4. Fine tune and host an existing, supported model with a different data set using SageMaker.

#### Instructor-led workshop
If you are participating in an instructor-led workshop, follow the guidance provided by your instructor for accessing the environment.

#### Self-managed workshop
If you are following the workshop steps in your own environment, you will need to take the following actions:
1. Launch a trn1.2xlarge instance on Amazon EC2, using the latest [DLAMI with Neuron packages preinstalled](https://repost.aws/articles/ARTxLi0wndTwquyl7frQYuKg) 
2. Use a Python virtual environment preinstalled in that DLAMI, commonly located in `/opt/aws_<xxx>`.
3. Set up and manage your own development environment on that instance, such as by using VSCode or a Jupyter Lab server.

### Background knowledge
This workshop introduces developing on AWS Trainium for the academic AI research audience and technical innovators. As such it's expected that the audience will already have a firm understanding of machine learning fundamentals. 

### Workshop costs
If you are participating in an instructor-led workshop hosted in an AWS-managed Workshop Studio environment, you will not incur any costs through using this environment. If you are following this workshop in your own environment, then you will incur associated costs with provisioning an Amazon EC2 instance. Please see the service pricing details [here](https://aws.amazon.com/ec2/pricing/on-demand/). 

At the time of writing, this workshop uses a trn1.2xlarge instance with an on-demand hourly rate in supported US regions of $1.34 per hour. The fine tuning workshop requires less than an hour of ml.trn1.2xlarge at $1.54 per hour, and an ml.inf2.xlarge at $0.99 per hour. Please ensure you delete the resources when you are finished.

## FAQ's and known issues
1. Workshop instructions are available [here](https://catalog.us-east-1.prod.workshops.aws/workshops/bf9d80a3-5e4b-4648-bca8-1d887bb2a9ca/en-US).
2. If you use the `NousResearch` Llama 3.2 1B, please note you'll need to remove a trailing comma in the model config file. You can do this by using VIM in VSCode. If you do not take this step, you'll get an error for invalid JSON in trying to read the model config in Lab 1. If editing the file through the terminal is a little challenging, you can also download the config file from this repository with the following command:
   `!wget https://raw.githubusercontent.com/aws-neuron/neuron-workshops/main/labs/NxD/generation_config.json -P /home/ec2-user/environment/models/llama/`
4. Jupyter kernels can hold on to the NeuronCores as a Python process even after your cell has completed. This can then cause issues when you try to run a new notebook, and sometimes when you try to run another cell. If you encounter a `NeuronCore not found` or similar error statement, please just restart your Jupyter kernel and/or shut down kernels from previous sessions. You can also restart the instance through the EC2 console. Once your node is back online, you can always check the availability of the NeuronCores with `neuron-ls`.
5. Want to see how to integrate NKI with NxD? Check out our `nki-llama` [here](https://github.com/aws-samples/nki-llama).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

