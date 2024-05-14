# GAMA-Code-Generator
------
## Overview 
Meet GAMA-GPT, a text-to-code model specifically engineered to produce high-quality GAML code snippets in response to user input prompts. 

Derived from Mistral-Instruct-v2.0 and fine-tuned for optimal performance, this model boasts a rapid response time, typically returning a question in under 3 seconds. Notably, GAMA-GPT surpasses other widely-known AI tools like ChatGPT, POE, Gemini, BingAI, and StarCoder in the quality of GAML code it generates.

------
## Purpose
This project **_aims to create a text-to-code model designed specifically for generating GAML language in response to user questions_**. 

------
Inspired by ChatGPT and other prominent AI tools, its objective is to facilitate the generation of code snippets tailored to the GAML language. **_This model strives to enhance the learning experience for newcomers to GAML programming by providing intuitive assistance._** Additionally, for users seeking to utilize GAML for their tasks without delving deeply into programming, this model offers a helpful solution for generating code.

![](https://github.com/ACROSS-Lab/GAMA-Code-Generator/blob/main/assets/comparison-img.png)

The right Figure above illustrates that when given a consistent prompt input, ChatGPT typically yields subpar responses specifically for the GAML language. Conversely, tools equipped with our fine-tuned for this purpose tend to produce significantly more accurate results.

## Implementation

### For Inference
Users can follow this instructions for performing inference.
#### 1. Download this project 
```
git clone https://github.com/ACROSS-Lab/GAMA-Code-Generator.git
```

#### 2. Create environment
We need an environment since some ertain specific libraries can only be installed within an environment.
Feel free to utilize either a Python environment or a Conda environment based on your preference.

  - **i. For Conda:**
          You can follow instructions from  [miniconda](https://docs.anaconda.com/free/miniconda/index.html) website to install Conda environment and [cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) to use it or follow below instruct for fast create and activate Conda environment:
```

conda create --name gamagpt-inference            # create environment named gamagpt-inference
conda activate gamagpt-inference            # activate conda environment
conda deactivate          # if you want to use other environments or just don't want to stay on this environment
``` 
  - **ii. For Python environment**, follow bellow instructs:
```
python -m venv gamagpt-chatbot
source gamagpt-inference/bin/activate #for ubuntu
gamagpt-inference/Scripts/activate #for windows
```
#### 3. Install libraries
Following the environment installation, activate your environment and install the required libraries by running:
```
cd path/to/directory/GAMA-Code-Generator
pip install -r requirements.txt
```
> [!NOTE]
> - We suggest setting up an environment to successfully complete this whole project, including fine-tuning the model with your own datasets and formatting it to GGUF format for building a web/app product.
> - Each part of the project will require installation of multiple libraries, and there may be conflicts between them.
> - Using environments will get rid of confliction between libraries.
#### 4. Run inference on your local machine
`bash
python /path/to/
`
