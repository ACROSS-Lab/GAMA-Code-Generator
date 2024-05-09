from selenium import webdriver
from selenium.webdriver.common.by import By
import json
import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from fake_useragent import UserAgent
from selenium.webdriver.support import expected_conditions as EC
from multiprocessing import Pool
import tqdm
from selenium.webdriver.support.ui import Select

nb_processes = 16
max_waiting_time = 1200 # 120s
# Providers: Auto, You, Bing, HuggingChat, Gemini, Phind, ...
# Models: Default, gpt-3.5-turbo, gemini-pro, ....
#setting = {"provider": "You", "model": "gpt-4"}
#setting = {"provider": "Auto", "model": "Default"}
setting = {"provider": "Bing", "model": "Default"}

def generate_prompt(gaml_code_snippet):
    try:
        prompt_template = '''
Act as a prompt generator. I give you a GAML (GAma Modeling Language) code snippet and you generate the 
corresponding prompt that can be used to ask Large Languague Model to create this GAML code snippet for me.
IIt is very important to remember that you should return only the prompt, no additional text. 
Therefore, your anwser must strictly start with "Here is the prompt:".
This is my GAML code snippet:
***
xxx
***
''' 
        gaml_code_template_integrated = prompt_template.replace('xxx', gaml_code_snippet)

        driver = webdriver.Firefox()
        driver.get('http://127.0.0.1:8080/chat/')
        sleep(1)

        if setting["provider"] != "Auto":
            select_provider = Select(driver.find_element(By.ID, 'provider'))
            select_provider.select_by_visible_text(setting["provider"])
            sleep(1)

        if setting["model"] != "Default":
            select_model = Select(driver.find_element(By.ID, 'model'))
            select_model.select_by_visible_text(setting["model"])
            sleep(1)

        # fill text
        input_element = driver.find_element(By.ID, "message-input")
        input_element.send_keys(gaml_code_template_integrated)
        sleep(2)

        check_successful_click = False

        while check_successful_click == False:
            check1 = driver.find_element(By.CLASS_NAME, "stop_generating-hidden")
            check2 = driver.find_element(By.CLASS_NAME, "regenerate-hidden")
            
            if check1 and check2:
                input_element = driver.find_element(By.ID, "send-button")
                input_element.click()
                sleep(1)
                check_successful_click = True

        # wait maximum 100s or until the element is clickable
        element = WebDriverWait(driver, max_waiting_time).until(EC.element_to_be_clickable((By.ID, "regenerateButton")))

        output_element = driver.find_element(By.CLASS_NAME, "content_inner")
        # get prompt in html
        prompt = output_element.get_attribute('innerHTML')

        driver.find_element(By.XPATH, "//button[@onclick=\"delete_conversations()\"]").click()

        sleep(1)

        driver.quit()

        return {"question":prompt, "answer": gaml_code_snippet}
    
    except Exception as e:
        print(e)
        return ''


file_name = '/home/phanh/Downloads/finetuneGAMA/data/train/json/ExampleQuery.json'
with open(file_name) as f:
    data = json.load(f)
    gaml_codes_list = data["data"]["prompts"]
    gaml_codes = [gaml_code["answer"] for gaml_code in gaml_codes_list]

print(generate_prompt(gaml_codes[0]))
