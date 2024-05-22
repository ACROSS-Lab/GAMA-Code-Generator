import torch
import gradio as gr
import multiprocessing




from textwrap import fill
from IPython.display import Markdown, display

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, UnstructuredMarkdownLoader
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, TextIteratorStreamer, StoppingCriteriaList

import warnings
from threading import Thread
from transformers import TextIteratorStreamer, StoppingCriteriaList

warnings.filterwarnings('ignore')

def main():

    def load_model_and_tokenizer(model_name):
        """Load the model and tokenizer with quantization configuration."""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config
        )

        return model, tokenizer

    def setup_generation_pipeline(model, tokenizer, model_name):
        """Setup the text generation pipeline with the specified generation configuration."""
        generation_config = GenerationConfig.from_pretrained(model_name)
        generation_config.max_new_tokens = 1024
        generation_config.temperature = 0.5
        generation_config.top_p = 0.95
        generation_config.top_k = 1000
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.15

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            generation_config=generation_config,
        )

    def load_embeddings(model_name):
        """Load HuggingFace embeddings with specified configurations."""
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
            multi_process=True,
            show_progress=True,
        )

    def load_articles(urls):
        """Load articles from the given URLs."""
        loader = UnstructuredURLLoader(urls=urls)
        return loader.load()

    def split_text(documents):
        """Split the documents into chunks using RecursiveCharacterTextSplitter."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        return text_splitter.split_documents(documents)

    def create_vectorstore(text_chunks, embeddings, persist_directory="db"):
        """Create a vector store from text chunks and embeddings."""
        return Chroma.from_documents(text_chunks, embeddings, persist_directory=persist_directory)

    def create_qa_chain(llm, retriever, memory, prompt_template):
        """Create a ConversationalRetrievalChain for question-answering."""
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            condense_question_prompt=prompt_template,
        )

    def generate_tokens(query, history, qa_chain, memory):
        """Generate tokens iteratively."""
        result = qa_chain({"question": query})
        message = result["answer"].strip()
        
        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                stop_ids = [29, 0]
                for stop_id in stop_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False


        # Setup token generation parameters
        stop = StopOnTokens()
        history_transformer_format = history + [[message, ""]]
        messages = "".join(["".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]]) for item in history_transformer_format])
        model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.95,
            top_k=1000,
            temperature=0.4,
            num_beams=1,
            stopping_criteria=StoppingCriteriaList([stop])
        )

        # Start generation
        for new_token in streamer:
            if new_token != '<' or '[':
                yield new_token

    def querying(query, history, qa_chain, memory):
        """Query the QA chain and return the generated tokens as a single string."""
        # Initialize an empty string to store the generated tokens
        generated_message = ""

        # Generate tokens iteratively
        for token in generate_tokens(query, history, qa_chain, memory):
            generated_message += f"```{token}```"

        return generated_message


    
    """Main function to set up and launch the Gradio interface."""
    MODEL_NAME = "Phanh2532/GAMA-Code-generator-v1.0"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    # Setup generation pipeline
    text_gen_pipeline = setup_generation_pipeline(model, tokenizer, MODEL_NAME)

    # Load embeddings
    embeddings = load_embeddings("thenlper/gte-large")

    # Load articles and split into chunks
    articles = [
        'https://gama-platform.org/wiki/Introduction',
        'https://gama-platform.org/wiki/StartWithGAML',
        'https://gama-platform.org/wiki/ModelOrganization',
        'https://gama-platform.org/wiki/BasicProgrammingConceptsInGAML',
        'https://gama-platform.org/wiki/ManipulateBasicSpecies',
        'https://gama-platform.org/wiki/GlobalSpecies',
        'https://gama-platform.org/wiki/RegularSpecies',
        'https://gama-platform.org/wiki/DefiningActionsAndBehaviors',
        'https://gama-platform.org/wiki/InteractionBetweenAgents',
        'https://gama-platform.org/wiki/AttachingSkills',
        'https://gama-platform.org/wiki/Inheritance'
    ]
    documents = load_articles(articles)
    text_chunks = split_text(documents)

    # Create vector store
    db = create_vectorstore(text_chunks, embeddings)

    # Setup LLM pipeline
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    # Define custom prompt template
    custom_template = """You are a GAML Code Generator AI assistant. You will generate GAML code snippet based on the question. If you do not know the answer, provide reply with 'I am sorry, I don't have enough information'.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:
        """
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    # Setup conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create QA chain
    qa_chain = create_qa_chain(llm, db.as_retriever(search_kwargs={"k": 3}), memory, CUSTOM_QUESTION_PROMPT)

    # Define Gradio interface
    iface = gr.ChatInterface(
        fn=lambda query, history: querying(query, history, qa_chain, memory),
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(placeholder="Generate a GAML model named 'city' which has 2 species named 'car' and 'bus'", container=False, scale=7),
        title="GAMABot",
        theme="soft",
        examples=["Generate a GAML model named 'city' which has 2 species named 'car' and 'bus'"],
        cache_examples=False,
        retry_btn="Retry",
        undo_btn="Undo",
        clear_btn="Clear",
        submit_btn="Submit"
    )
    iface.launch(share=True)

if __name__ == '__main__':
    main()

