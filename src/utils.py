from datetime import datetime
import base64
import yaml
import requests
from dotenv import load_dotenv
import streamlit as st
import os
import aiohttp
import asyncio
import time
load_dotenv()


def load_config(file_path = "config.yaml"):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    
config = load_config()


def convert_ns_to_seconds(ns_value):
    return ns_value / 1_000_000_000 

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def command(user_input):
    splitted_input = user_input.split(" ")
    if splitted_input[0] == "/pull":
        return pull_model_in_background(splitted_input[1])
    elif splitted_input[0] == "/help":
        return "Possible commands:\n- /pull <model_name>"
    else:    
        return """Invalid command, please use one of the following:\n
                    - /help\n
                    - /pull <model_name>"""

def pull_ollama_model(model_name):
    json_response = requests.post(url = config["ollama"]["base_url"] + "/api/pull", json = {"model": model_name}).json()
    print(json_response)
    if "error" in json_response.keys():
        return json_response["error"]["message"]
    else:
        st.session_state.model_options = list_ollama_models()
        st.warning(f"Pulling {model_name} finished.")
        return json_response

async def pull_ollama_model_async(model_name, stream=True, retries=1):
    url = config["ollama"]["base_url"] + "/api/pull"
    json_data = {"model": model_name, "stream": stream}
    
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:  # Increased timeout for model pulling
                async with session.post(url, json=json_data) as response:
                    if stream:
                        # Handle streaming response
                        async for chunk in response.content.iter_chunked(1024):
                            if chunk:
                                st.info(f"Received chunk: {chunk.decode('utf-8')}")
                    else:
                        json_response = await response.json()
                        print(json_response)

                        if json_response.get("error", False):
                            return json_response["error"]
                        else:
                            st.session_state.model_options = list_ollama_models()
                            return f"Pull of {model_name} finished."
                    return "Pulled"
        except asyncio.TimeoutError:
            st.warning(f"Timeout on attempt {attempt + 1}. Retrying...")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            break  # Break on non-timeout errors
    return f"Failed to pull {model_name} after {retries} attempts."

# Function to trigger the async pull (can be run in the event loop)
def pull_model_in_background(model_name, stream=False):
    try:
        # Check if there's already an event loop running
        loop = asyncio.get_running_loop()
    except RuntimeError:  # If no loop is running, start a new one
        loop = None

    #st.info(f"Pulling {model_name}...")

    if loop and loop.is_running():
        # If an event loop is already running, create a task for the async function
        return asyncio.create_task(pull_ollama_model_async(model_name, stream=stream))
    else:
        # Otherwise, use asyncio.run() to run it synchronously
        return asyncio.run(pull_ollama_model_async(model_name, stream=stream))



def list_openai_models():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    response = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {openai_api_key}"}).json()
    if response.get("error", False):
        st.warning("Openai Error: " + response["error"]["message"])
        return []
    else:
        return [item["id"] for item in response["data"]]


def list_ollama_models():
    json_response = requests.get(url = config["ollama"]["base_url"] + "/api/tags").json()
    if json_response.get("error", False):
        return []
    models = [model["name"] for model in json_response["models"] if "embed" not in model["name"]]
    return models
    
def convert_bytes_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")
    
def convert_bytes_to_base64_with_prefix(image_bytes):
    return "data:image/jpeg;base64," + convert_bytes_to_base64(image_bytes)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_avatar(sender_type):
    if sender_type == "user":
        return "chat_icons/user_image.png"
    else:
       return "chat_icons/bot_image.png"

def save_config(config):
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)