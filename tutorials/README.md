# FINETUNE TUTORIAL
![_Main pipeline_](https://github.com/ACROSS-Lab/GAMABot/blob/main/assets/rlhf-pipeline.png)

## 1. Create your own dataset
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
## 2. Finetune Model
Run this in your terminal
```
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

## 3. Run your brand new model
### 3.1. Fast inference
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

### 3.2. Gradio app
Edit `./app/gradio-app.py` and change the `model_id` with your HuggingFace repo_id of your model and then run it and you have your chatbot now.
