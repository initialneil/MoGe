import os
import os.path as osp
import tyro
from dataclasses import dataclass
from openai import OpenAI 
import base64
import json
from pydantic import BaseModel
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from textwrap import dedent
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from collov.files import scandir
from collov.jsonl import read, write

########################################
import concurrent.futures

def parallel_foreach(_func, _args_list, max_workers=8, show_tqdm=False):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executrer:
        res_list = []
        if show_tqdm:
            from tqdm import tqdm
            for res in tqdm(executrer.map(_func, _args_list), total=len(_args_list)):
                res_list.append(res)
        else:
            for res in executrer.map(_func, _args_list):
                res_list.append(res)
    return res_list

########################################
# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Define your desired output schema
class CheckItem(BaseModel):
    describe: str   # Description of the invalid change
    passed: bool    # True: pass the check, False: failed

class CheckResponse(BaseModel):
    window_check: CheckItem
    door_check: CheckItem
    ceiling_check: CheckItem
    wall_check: CheckItem
    embed_closet_check: CheckItem
    kitchen_island_check: CheckItem
    fireplace_check: CheckItem
    all_passed: bool

INSTRUCTION = dedent("""\
    The first image shows an empty room. 
    The second image shows the same room with some furniture, decoration, painting, items, etc. 
    The goal is to check whether the building structure and non-removable furniture is valid. 
    Pass:
        1. the location of building structure and non-removable furniture is the same in both images.
        2. appearance change is allowed.
        3. painting or decoration is allowed.
        3. adding personal items is allowed.
        5. opening or closing doors is allowed.
        6. adding or removing movable furniture is allowed.
    Fail:
        1. adding or removing windows or doors is invalid.
        2. changing the bearing wall structure is invalid.
        3. adding or removing non-movable wall-embedded closet is invalid.
        4. adding or removing non-movable kitchen island is invalid.
        5. adding or removing fireplace is invalid.
                     
    If fail, describe the difference in the first image and the change in the second image. 
    For example: 
        window_check: 
            describe: the first image has no window on the right of the wall. the second image added a new window on the right of the wall.
            passed: False
        embed_closet_check:
            describe: the first image has a non-movable wall-embedded closet on the left with door closed. the second image has a non-movable wall-embedded closet on the left with door open.
            passed: True
        wall_check:
            describe: no change
            passed: True
        wall_check:
            describe: the first image has a bearing wall in the middle. the second image removed the bearing wall in the middle.
            passed: False
        wall_check:
            describe: the second image has new drawing on the wall
            passed: True
        wall_check:
            describe: the second image added a tv and a painting on the wall
            passed: True
        fireplace_check:
            describe: both images do not have fireplace
            passed: True
        ceiling_check:
            describe: the second image added a ceiling fan.
            passed: False
        ceiling_check:
            describe: the second image changed the appearance of the ceiling lamp.
            passed: True
        ceiling_check:
            describe: the second image changed the ceiling opening to a light.
            passed: False
        kitchen_island_check:
            describe: the room is not kitchen
            passed: True
        kitchen_island_check:
            describe: the second image added a movable tea table.
            passed: True
    Check: 
    1. window_check: whether the window locations are the same. add or remove windows is not allowed. 
    2. door_check: whether the door locations are the same. add or remove doors is not allowed. opening or closing doors is allowed. 
    3. ceiling_check: whether the ceilings and ceiling items (ceiling fans, ceiling lamps, ceiling openings) are the same. 
    4. wall_check: whether the wall structures are the same. change of wall structures is not allowed. 
    5. embed_closet_check: whether the non-movable wall-embedded closets are the same. add or remove non-movable wall-embedded closets is not allowed. 
    6. kitchen_island_check: whether the kitchen islands are the same. add or remove kitchen islands is not allowed. 
    7. fireplace_check: whether the fireplaces are the same. add or remove fireplaces is not allowed. 
    Finally, give an overall 'all_pass' boolean result.
    IMPORTANT: invalid change must be described.
    IMPORTANT: valid changes can be omitted.
    IMPORTANT: always generate the description first to help thinking, and then the check result.\
""")

def vlm_check(image_path, empty_path, model="gpt-5"):
    ## Set the API key and model name
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>"))

    base64_image = encode_image(image_path)
    base64_empty = encode_image(empty_path)

    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about images."},
            {"role": "user", "content": [
                {"type": "text", "text": INSTRUCTION},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_empty}"
                }},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }},
            ]}
        ],
        response_format=CheckResponse,
    )

    # Extract usage information
    usage = completion.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    
    # GPT pricing https://platform.openai.com/docs/pricing
    pricing = {
        "gpt-5": {
            "input_per_1K_tokens": 1.25,
            "output_per_1K_tokens": 10.00,
        },
        "gpt-5-mini": {
            "input_per_1K_tokens": 0.25,
            "output_per_1K_tokens": 2.00,
        }
    }

    input_cost = (prompt_tokens / 1_000_000) * pricing[model]["input_per_1K_tokens"]
    output_cost = (completion_tokens / 1_000_000) * pricing[model]["output_per_1K_tokens"]
    total_cost = input_cost + output_cost

    print(f'Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}')
    print(f'Cost - Input: ${input_cost:.6f}, Output: ${output_cost:.6f}, Total: ${total_cost:.6f}')

    message = completion.choices[0].message
    if message.parsed:
        print('--------------------------------------------------')
        print(f'{image_path}')
        print(f"{message.parsed}")
        result = message.parsed.model_dump()
        # Add usage stats to result
        result['_usage'] = {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        }
        return result
    else:
        print(message.refusal)
        return None

# Global variables to track total usage
total_usage_stats = {
    'total_images': 0,
    'total_prompt_tokens': 0,
    'total_completion_tokens': 0,
    'total_tokens': 0,
    'total_cost': 0.0
}

def run_vlm_check(func_args):
    global total_usage_stats
    
    image_path = func_args['image']
    empty_path = func_args['empty']
    check_fn = func_args['check_fn']
    model = func_args['model']
    if osp.exists(check_fn):
        print(f'check exists: {check_fn}')
        return
    
    msg = vlm_check(image_path, empty_path, model=model)
    if msg is not None:
        # Update global stats
        if '_usage' in msg:
            usage = msg['_usage']
            total_usage_stats['total_images'] += 1
            total_usage_stats['total_prompt_tokens'] += usage['prompt_tokens']
            total_usage_stats['total_completion_tokens'] += usage['completion_tokens']
            total_usage_stats['total_tokens'] += usage['total_tokens']
            total_usage_stats['total_cost'] += usage['total_cost']
        
        os.makedirs(osp.dirname(check_fn), exist_ok=True)
        with open(check_fn, 'w') as fp:
            json.dump(msg, fp, indent=4)

@dataclass
class Args:
    data_root: str
    input_json_fn: str
    output_json_fn: str
    model: str

def main():
    args = tyro.cli(Args)
    def to_abs_path(fn):
        return fn if osp.isabs(fn) else osp.join(args.data_root, fn)
    input_json_fn = to_abs_path(args.input_json_fn)
    output_json_fn = to_abs_path(args.output_json_fn)

    check_dir = osp.join(args.data_root, 'check', f'check_vlm_v1_{args.model}')
    os.makedirs(check_dir, exist_ok=True)

    data_list = read(input_json_fn)
    func_args_list = []

    for item in data_list:
        image_path = to_abs_path(item['image'])
        empty_path = to_abs_path(item['empty'])
        check_fn = osp.join(check_dir, item['image'] + '.json')
        if osp.exists(check_fn):
            print(f'check exists: {check_fn}')
            continue

        func_args_list.append({
            'image': image_path,
            'empty': empty_path,
            'check_fn': check_fn,
            'model': args.model,
        })

    all_infos = parallel_foreach(run_vlm_check, func_args_list,
                                 max_workers=8, show_tqdm=True)

    # Print total usage summary
    print("\n" + "="*50)
    print("TOTAL USAGE SUMMARY")
    print("="*50)
    print(f"Total images processed: {total_usage_stats['total_images']}")
    print(f"Total prompt tokens: {total_usage_stats['total_prompt_tokens']:,}")
    print(f"Total completion tokens: {total_usage_stats['total_completion_tokens']:,}")
    print(f"Total tokens: {total_usage_stats['total_tokens']:,}")
    print(f"Total cost: ${total_usage_stats['total_cost']:.4f}")
    if total_usage_stats['total_images'] > 0:
        avg_cost = total_usage_stats['total_cost'] / total_usage_stats['total_images']
        print(f"Average cost per image: ${avg_cost:.4f}")

    pass


if __name__ == '__main__':
    main()

