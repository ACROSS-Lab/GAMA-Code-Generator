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
conda activate gamagpt-inference            # activate conda environment
conda deactivate          # if you want to use other environments or just don't want to stay on this environment
``` 
  - **ii. For Python environment**, follow bellow instructs:
```
python -m venv gamagpt-chatbot
source gamagpt-inference/bin/activate       # for ubuntu
gamagpt-inference/Scripts/activate       # for windows
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
You can have 2 options
- **i. Run straight on your local machine**
```
python ./inference/python/command-line-inference.py
```
- **ii. Or you can run it on Google Colab/Jupyter Notebook** by following each steps in file named `peft-inference.ipynb` in `./inference/ipynb` directory. 

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
> - *For more detailed information about the pipeline, each stage, and in-depth knowledge about how it works, please follow the tutorials in the `tutorials` directory.*
> - Or you can follow these steps down below to do a fast fine-tuned with `Mistral-Instruct-7B-v2.0` model.
> - Please keep in mind this part will only show how a fast way to finetune Mistral without apply DPO training on it.

> Before proceeding, follow the `Create environment` instructions to create a new environment for fine-tuning and follow below instructions
> ```
> cd /path/to/GAMABot/train/
> pip install -r requirements.txt        
> ```
> Now you're all good! Let's finetune model to your preference

### 4.1. Create your own dataset
Our based model is `Mistral-Instruct-7B-v0.2` so we will follow their Instruction format for our dataset.

Due to MistralAI, in order to leverage instruction fine-tuning, your prompt should be surrounded by `[INST]` and `[/INST]` tokens. The very first instruction should begin with a begin of sentence id. The next instructions should not. The assistant generation will be ended by the end-of-sentence token id.


```
FOR EXAMPLE
PROMPT:
    "<s>[INST] What is your favourite condiment? [/INST]"

ANSWER:
    "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
```

So the idea is to create a dataset with pairs of questions - answers (however you like it, do it by yourself or take a public dataset) under a `.csv` format (which in my personal experience, is much handier) and then
```
python ./data/code/format-data.py
python ./data/code/csv2json.py
```
And then put your output to `data/train` or `data/eval` directory. And Voilá, you have your dataset to finetune model now!

------
### 4.2. Finetune Model
Run this in your terminal
```
cd /path/to/GAMABot
chmod u+x ./finetune.sh         # automatically train Mistral-Instruct-7B-v0.2 on 1 gpu
./finetune.sh
```
or
```
chmod u+x ./finetune-multi-gpus.sh                       
./finetune-multi-gpus.sh        # automatically train Mistral-Instruct-7B-v0.2 on multiple gpus
```
By running the bash script files, you have already trained the model using the parameters I provided. These parameters were set up as follows in `finetune.sh` and `finetune-multi-gpus.sh`:
```py
BASE_MODEL_ID="Mistral-Instruct-7B-v2.0"        # change this with HuggingFace repo id of your base model (could be LLama/Phi/etc.)
LOAD_IN_4BIT=true                               # change this to false if you don't want to apply QLoRA while finetuning
BNB_4BIT_USE_DOUBLE_QUANT=True                  
BNB_4BIT_QUANT_TYPE="nf4"
BNB_4BIT_COMPUTE_DTYPE="bfloat16"
TRAIN_DATA_PATH="/path/to/your/train-data.json"    
EVAL_DATA_PATH="/path/to/your/eval-data.json"
WANDB_PROJECT="gamaft-total"            # change this if you don't want to use wanb 
OUTPUT_DIR="/path/to/your/output/dir/"
WARMUP_STEPS=1
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_GPU_TRAIN_BATCH_SIZE=2
AUTO_FIND_BATCH_SIZE=true
GRADIENT_ACCUMULATION_STEPS=1
GRADIENT_CHECKPOINTING=True
MAX_STEPS=5
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=2e-5
FP16=true
OPTIM="paged_adamw_8bit"
LOGGING_STEPS=50
LOGGING_DIR='./logs'        
SAVE_STRATEGY="steps"
SAVE_STEPS=5
EVALUATION_STRATEGY="steps"    
EVAL_STEPS=50
DO_EVAL=true
OVERWRITE_OUTPUT_DIR=false         
REPORT_TO="wandb"        # comment this if you don't want to use wandb
```
To know better about how parameters and its values will effect our model, please visit [HuggingFace Trainer Parameters Documentation](https://huggingface.co/docs/transformers/main_classes/trainer) for more details.

### 5. Run your brand new model
#### 5.1. Fast inference
Copy and run these python code OR run `inference/command-line-inference.py` file (Remember to change the path of your model) to test your new model!
```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
base_model_id = "mistralai/Mistral-Instruct-7B-v0.2"    # or path to your HuggingFace repo_id of your base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)
eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
```
and then
```py
from peft import PeftModel
import torch
ft_model = PeftModel.from_pretrained(base_model, "/path/to/your/local/finetuned/model")
eval_prompt = "Create a GAML code snippet inspired by water pollution in real life"
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")
ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=2000, repetition_penalty=1.15)[0], skip_special_tokens=True))
    print('----------------------------------------------------------------------')
    #print(eval_tokenizer.decode(ft_model2.generate(**model_input, max_new_tokens=2000, repetition_penalty=1.15)[0], skip_special_tokens=True))
```

The output should be somethink like this
```
Create a GAML model name air_pollution to simulate air pollution in reality. The model should include species such as cars, trucks, buses, and factories that emit pollutants into the air. Pollutants should be spread throughout the environment based on their emission rates from these species. Additionally, people can be introduced into the model to represent those affected by air pollution.

experiment air_pollution type: gui {
    parameter "Initial number of cars: " var: nb_cars_init min: 1 max: 1000 category: "Cars";
    output {
        display main_display {
            species cars aspect: base;
            species people aspect: base;
        }
    }
}
....
```

#### 5.2. Gradio app
Edit `./app/gradio-app.py` and change the `model_id` with your HuggingFace repo_id of your model and then run it and you have your chatbot now.
