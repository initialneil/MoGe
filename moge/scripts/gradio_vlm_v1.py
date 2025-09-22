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
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from collov.files import scandir
from collov.jsonl import read, write
import gradio as gr

@dataclass
class Args:
    data_root: str
    input_json_fn: str
    model: str

def main():
    args = tyro.cli(Args)
    def to_abs_path(fn):
        return fn if osp.isabs(fn) else osp.join(args.data_root, fn)
    input_json_fn = to_abs_path(args.input_json_fn)

    check_dir = osp.join(args.data_root, 'check', f'check_vlm_v1_{args.model}')
    os.makedirs(check_dir, exist_ok=True)

    data_list = read(input_json_fn)
    pass_list = []
    failed_list = []

    for item in data_list:
        image_path = to_abs_path(item['image'])
        empty_path = to_abs_path(item['empty'])
        print('--------------------------------------------------')
        print(f'{image_path}')

        check_fn = osp.join(check_dir, item['image'] + '.json')
        if osp.exists(check_fn):
            with open(check_fn, 'r') as fp:
                cc = json.load(fp)

            if cc.get('all_passed', False):
                pass_list.append({
                    'image': item['image'],
                    'empty': item['empty'],
                    'caption': cc,
                })
            else:
                failed_list.append({
                    'image': item['image'],
                    'empty': item['empty'],
                    'caption': cc,
                })
                pass

    ##################################################
    # show failed_list with gradio
    
    # File to persist current index across refreshes
    state_file = osp.join(check_dir, 'gradio_state.json')
    
    def load_saved_index():
        """Load the last saved index from file"""
        if osp.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                return state.get('current_index', 0)
            except:
                return 0
        return 0
    
    def save_current_index(index):
        """Save current index to file"""
        try:
            with open(state_file, 'w') as f:
                json.dump({'current_index': index}, f)
        except:
            pass

    def update_display(index, current_state):
        if not failed_list or index < 0 or index >= len(failed_list):
            return None, None, "", gr.update(maximum=max(0, len(failed_list)-1)), current_state
        
        item = failed_list[index]
        image_path = to_abs_path(item['image'])
        empty_path = to_abs_path(item['empty'])
        
        # Filter caption to only show checks where describe != "no change"
        caption_data = item['caption']
        if isinstance(caption_data, dict):
            caption_text = json.dumps(caption_data, indent=2)
        else:
            caption_text = str(caption_data)
        
        # Update the state with current index
        new_state = index
        # Save to file for persistence across refreshes
        save_current_index(index)
        
        return image_path, empty_path, caption_text, gr.update(maximum=max(0, len(failed_list)-1)), new_state

    def update_from_slider(slider_value, current_state):
        return update_display(int(slider_value), current_state)

    def update_from_number(number_value, current_state):
        return update_display(number_value, current_state)

    # Create Gradio interface
    with gr.Blocks(title=f"VLM Check Results [{args.model}]") as demo:
        # Load saved index from file
        saved_index = load_saved_index()
        saved_index = max(0, min(saved_index, len(failed_list) - 1)) if failed_list else 0
        
        # State to track current index
        current_index_state = gr.State(value=saved_index)
        
        gr.Markdown(f"# VLM Check Results ({len(failed_list)} items)")
        
        # First row: side by side images
        with gr.Row():
            empty_display = gr.Image(label="Empty Image", type="filepath", height=500)
            image_display = gr.Image(label="Original Image", type="filepath", height=500)
        
        # Second row: caption text
        with gr.Row():
            caption_display = gr.Textbox(
                label=f"Caption: [{args.model}]",
                lines=10,
                max_lines=20,
                interactive=False
            )
        
        # Third row: navigation controls
        with gr.Row():
            with gr.Column(scale=1):
                prev_button = gr.Button("◀ Previous", size="sm")
            with gr.Column(scale=6):
                index_slider = gr.Slider(
                    minimum=0,
                    maximum=max(0, len(failed_list)-1),
                    step=1,
                    value=saved_index,
                    label="Item Index"
                )
            with gr.Column(scale=1):
                next_button = gr.Button("Next ▶", size="sm")
            with gr.Column(scale=2):
                index_number = gr.Number(
                    label="Go to Index",
                    value=saved_index,
                    minimum=0,
                    maximum=max(0, len(failed_list)-1),
                    precision=0
                )

        # Event handlers
        def prev_item(current_state):
            new_index = max(0, current_state - 1)
            return new_index, *update_display(new_index, current_state)

        def next_item(current_state):
            new_index = min(len(failed_list) - 1, current_state + 1)
            return new_index, *update_display(new_index, current_state)

        def load_from_state(current_state):
            # When page loads, use the stored state index
            return current_state, *update_display(current_state, current_state)

        index_slider.change(
            fn=update_from_slider,
            inputs=[index_slider, current_index_state],
            outputs=[image_display, empty_display, caption_display, index_slider, current_index_state]
        )
        
        index_number.submit(
            fn=update_from_number,
            inputs=[index_number, current_index_state],
            outputs=[image_display, empty_display, caption_display, index_slider, current_index_state]
        )

        prev_button.click(
            fn=prev_item,
            inputs=[current_index_state],
            outputs=[index_slider, image_display, empty_display, caption_display, index_slider, current_index_state]
        )

        next_button.click(
            fn=next_item,
            inputs=[current_index_state],
            outputs=[index_slider, image_display, empty_display, caption_display, index_slider, current_index_state]
        )

        # Initialize with stored state on page load
        if failed_list:
            demo.load(
                fn=load_from_state,
                inputs=[current_index_state],
                outputs=[index_slider, image_display, empty_display, caption_display, index_slider, current_index_state]
            )

    # Launch the interface
    demo.launch(
        share=False, 
        server_name="0.0.0.0", 
        server_port=8080,
        allowed_paths=["/home/szj/sdc/data/nb_empty_syn"]
    )


if __name__ == '__main__':
    main()

