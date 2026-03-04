from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(  api_key= api_key)

prompt_prefix = f"you are a function. I will give a prompt in the form \
    FUNCTION_NAME, [x1,x2,...,xn] and you will respond with [y1,y2,...yn] based on\
    the appropriate output for the function of name FUNCTION_NAME.\
    Do not add any white space, simply return the values in the same form you received them unless it's an nd.array, then convert it to a list\
    If you receieve multiple inputs, the variables will be listed in the same order as in the equation. \
        PROMPT: "
SIGMOID_PROMPT = lambda inputs: prompt_prefix + f"SIGMOID, {list(inputs)}" 



def call_to_api(prompt:str) -> str:
    """ 
    inputs prompt to gpt and returns the response. 
    """
    response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Generate"}
    ],
    store=True,
    )
    try:
        ret = response.choices[0].message.content or "-1"
    except:
        ret = "-1"
    finally:
        return ret
    
def gpt_func(fn:str, x):
    """
    Return f(x) using gpt.
    """
    prompt = prompt_prefix + fn + ', ' + str(x)
    response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Generate"}
    ],
    store=True,
    )
    try:
        ret = response.choices[0].message.content or "-1"
    except:
        ret = "-1"
    finally:
        print(fn, x, ret)
        return ret
    
