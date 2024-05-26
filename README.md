<p align="center">
    <br>
    <img src="https://github.com/ACROSS-Lab/GAMABot/blob/main/assets/logo.png" width="1000" height="150"/>
    <br>
</p>


--------
> [!NOTE]
> - This project is built by fine-tuning [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) on a GAML Code dataset created by members of ACROSS-Lab
> - Curently there are 2 version of model you can choose to deploy your own GAMABot
>   - 🤗 [**GAMA-Code-Generator-v1.0**](https://huggingface.co/Phanh2532/GAMA-Code-generator-v1.0) - finetuned Mistral-7B-Instruct-v0.2 on Finetune Dataset
>   - 🤗 [**GAMA-Code-Generator-v2.0**](https://huggingface.co/Phanh2532/GAMA-Code-generator-v2.0) - trained with DPO - a novel approach which share the same idea with Reinforcement Learning Human Feedback on Reward Dataset
>   - 🚀 [**Finetune Dataset**](https://huggingface.co/datasets/Phanh2532/GAML-Data) - 520 pairs of question - answer about GAML language
>   - 🚀 [**Reward Dataset**](https://huggingface.co/datasets/Phanh2532/reward-GAML) - 149 pairs of question - answer about GAML language

> [!IMPORTANT]
>   - We _strongly suggest creating a HuggingFace account and log in using `huggingface-cli login` for this project_. It can greatly streamline the process later on as our models and datasets are publicly accessible on HuggingFace. But still, this is optional.
>   - For people who don't want to use HuggingFace platform:
>       - To deploy GAMABot, please visit via upper link and download version of GAMA-Code-Generator you like to your local machine.    
>       - To finetune Mistral with your own dataset, please follow instructions from [Mistral AI Repo](https://github.com/mistralai/mistral-inference/tree/main) to download Mistral as your base model.

--------
[![🚀**Little Demo with GAMAChatbot**](https://github.com/ACROSS-Lab/GAMA-Code-Generator/blob/main/assets/DemoGAMABOT.png)](https://www.youtube.com/watch?v=7m-WpGrlJ0U)

--------
## 1. Overview 
Meet GAMA-GPT, a chatbot built from text-to-code model specifically engineered to produce high-quality GAML code snippets in response to user input prompts. 

Derived from Mistral-Instruct-v2.0 and fine-tuned for optimal performance, this model boasts a rapid response time, typically returning a question in under 3 seconds. Notably, GAMA-GPT surpasses other widely-known AI tools like ChatGPT, POE, Gemini, BingAI, and StarCoder in the quality of GAML code it generates.


## 2. Purpose
This project **_aims to create a text-to-code model designed specifically for generating GAML language in response to user questions and a friendly interface for users_**. 

Inspired by ChatGPT and other prominent AI tools, its objective is to facilitate the generation of code snippets tailored to the GAML language. **_This model strives to enhance the learning experience for newcomers to GAML programming by providing intuitive assistance._** Additionally, for users seeking to utilize GAML for their tasks without delving deeply into programming, this model offers a helpful solution for generating code.

![](https://github.com/ACROSS-Lab/GAMA-Code-Generator/blob/main/assets/comparison-img.png)

The right Figure above illustrates that when given a consistent prompt input, ChatGPT typically yields subpar responses specifically for the GAML language. Conversely, tools equipped with our fine-tuned for this purpose tend to produce significantly more accurate results.

------
Users can follow this instructions for performing inference or run your chatbot interface.

### 3.1. Requirements
#### 3.1.1. Download this project 
```
git clone https://github.com/ACROSS-Lab/GAMABot.git
```

#### 3.1.2. Create environment
We need an environment since some ertain specific libraries can only be installed within an environment.
Feel free to utilize either a Python environment or a Conda environment based on your preference.

  - **i. For Conda:**
          You can follow instructions from  [miniconda](https://docs.anaconda.com/free/miniconda/index.html) website to install Conda environment and [cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) to use it or follow below instruct for fast create and activate Conda environment:
```
conda create --name gamagpt-inference            # create environment named gamagpt-inference (change the name to your preference)
conda activate gamagpt-inference                 # activate conda environment
conda deactivate                                 # if you want to use other environments or just don't want to stay on this environment
``` 
  - **ii. For Python environment**, follow bellow instructs:
```
python -m venv gamagpt-chatbot
source gamagpt-inference/bin/activate       # for ubuntu
gamagpt-inference/Scripts/activate          # for windows
```
#### 3.1.3 Install libraries
Following the environment installation, activate your environment and install the required libraries by running:
```
cd path/to/GAMABot
pip install -r requirements.txt
```
> [!IMPORTANT]
> - We suggest setting up an environment to successfully complete this whole project, including fine-tuning the model with your own datasets and formatting it to GGUF format for building a web/app product.
> - Each part of the project will require installation of multiple libraries, and there may be conflicts between them.
> - Using environments will get rid of confliction between libraries.
------
### 3.2. Run inference
You can have 3 options
- **i. Copy and run this python program**
```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
model_id = "Phanh2532/GAMA-Code-generator-v1.0"   
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config,  
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)
eval_tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=True, trust_remote_code=True)

eval_prompt = "Create a GAML code snippet inspired by water pollution in real life" # Change this with your wanted prompt

model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=512, repetition_penalty=1.15)[0], skip_special_tokens=True))
    print('----------------------------------------------------------------------')
```
- **ii. Run python script**
This will generate output for bunch of prompts in `./inference/input.txt`. You can change the prompt by edit `input.txt` file.
```
python ./inference/python/command-line-inference.py
```
- **iii. Or you can run it on Google Colab/Jupyter Notebook** by following each steps in file named `peft-inference.ipynb` in `./inference/ipynb` directory. 

------
### 3.3. Run Chatbot Interface
- **i. Run straight on your local machine**
```
python ./app/gradio-app.py
```
- **ii. Or you can run it on Google Colab/Jupyter Notebook** by following each steps in file named `gradio-inference.ipynb` or `prompting-format-gradio-interface.ipynb` in `./inference/ipynb` directory. 

------


## 4. Finetuning use your own dataset
> [!NOTE]
> - For more detailed information about the pipeline, each stage, and in-depth knowledge about how it works, please follow the tutorials in the `tutorials` directory or via [this link](https://github.com/ACROSS-Lab/GAMABot/tree/main/tutorials).


> Before proceeding, follow the `Create environment` instructions to create a new environment for fine-tuning and follow below instructions
> ```
> cd /path/to/GAMABot/train/
> pip install -r requirements.txt        
> ```
> Now you're all good! Let's finetune model to your preference

------
## 5. Acknowledgement
- [Mistral 7B](https://arxiv.org/pdf/2310.06825) 
- [FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS](https://arxiv.org/pdf/2109.01652)
- [Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment](https://arxiv.org/pdf/2312.12148)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [Reinforcement Learning with Human Feedback: Learning Dynamic Choices via Pessimism](https://arxiv.org/pdf/2305.18438)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)

