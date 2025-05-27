# From Individuals to Interactions: Benchmarking Gender Bias in Multimodal Large Language Models from the Lens of Social Relationship

**Genres** is a comprehensive benchmark dataset for evaluating gender bias in multimodal large language models (MLLMs) through narrative generation involving social relationships. Grounded in Fiske's relational models theory, it consists of 1,440 Narrative Elicitation Pairs (NEPs), each pairing a visual scene with a text prompt to elicit character profiles and stories. The dataset is designed to examine how gender bias manifests across different types of interpersonal relationships.

## Table of Contents
- [Dataset](#the-dataset)
- [Codebase](#the-codebase)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [MLLMs Response Collection](#mllms-responses-collecting)
  - [Response Analysis](#responses-analysis)
  - [Bias Evaluation](#bias-evaluation)
- [License](#license)

## The Dataset

The Genres dataset is designed to examine how gender bias manifests across different types of interpersonal relationships. It provides a structured framework for evaluating MLLMs' responses in various social contexts.

The dataset is available at: [Hugging Face Datasets](https://huggingface.co/datasets/Savannah-y7/Genres)

## The Codebase

The Genres codebase is organized into three main components:

1. **[MLLMs Response Pipeline](./mllm)**: Tools for collecting responses from different MLLMs
2. **[Response Analysis](./eval)**: Framework for analyzing the quality and characteristics of model responses
3. **[Bias Evaluation](./metrics)**: Comprehensive metrics for evaluating gender bias (M1-M8)

## Installation

To set up the environment:

```bash
# Clone the repository
git clone https://github.com/Savannah2000/Genres.git
cd Genres

# Create and activate conda environment
conda create -n Genres python=3.10
conda activate Genres

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

1. Download the [Genres dataset](https://huggingface.co/datasets/Savannah-y7/Genres)
2. Store the dataset in the `./data` folder

### MLLMs Response Collection

The codebase supports inference for four models:
- Qwen2.5-VL-3B
- Phi-4-Multimodal
- Qwen2.5-VL-7B
- Janus-Pro-7B

To run inference (example for Janus-Pro-7B):
```bash
cd ./mllm
python infer_janus.py
```

Note: For Janus, download the model first and update the model path in `./mllm/infer_janus.py`.

Results will be saved in `./response/{modelname}` as JSONL files.

### Response Analysis

To analyze model responses:
```bash
cd ./eval
bash run_eval.sh
```

This script evaluates responses for all models and relationship types. You can modify the parameters to analyze specific models.

Analysis results are saved in `./analysis/{modelname}` as JSONL files.

### Bias Evaluation

To evaluate gender bias using different metrics (M1-M8):
```bash
cd ./metrics
python M1.py  # Run individual metrics as needed
```


## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC 4.0) - see the [LICENSE](LICENSE) file for details.

The CC-BY-NC 4.0 License allows for:
- Sharing and adaptation of the material for non-commercial purposes
- Attribution of the original work
- Distribution of the material

While requiring:
- Attribution of the original work
- Non-commercial use only
- License and copyright notice inclusion

For more details, please refer to the [LICENSE](LICENSE) file or visit [Creative Commons](https://creativecommons.org/licenses/by-nc/4.0/).
