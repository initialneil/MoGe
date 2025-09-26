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
    model: str  # Can be comma-separated list of models
    port: int = 8080  # Port for Gradio server

def main():
    args = tyro.cli(Args)
    def to_abs_path(fn):
        return fn if osp.isabs(fn) else osp.join(args.data_root, fn)
    input_json_fn = to_abs_path(args.input_json_fn)

    # Parse multiple models
    models = [m.strip() for m in args.model.split(',')]
    is_multi_model = len(models) > 1

    # Load data for all models
    all_model_data = {}
    data_list = read(input_json_fn)
    
    for model in models:
        check_dir = osp.join(args.data_root, 'check', f'check_vlm_v1_{model}')
        os.makedirs(check_dir, exist_ok=True)
        
        model_results = {}
        for item in data_list:
            check_fn = osp.join(check_dir, item['image'] + '.json')
            if osp.exists(check_fn):
                with open(check_fn, 'r') as fp:
                    cc = json.load(fp)
                model_results[item['image']] = {
                    'image': item['image'],
                    'empty': item['empty'],
                    'caption': cc,
                    'all_passed': cc.get('all_passed', False)
                }
        all_model_data[model] = model_results

    # Build display list based on mode
    if is_multi_model:
        # For multi-model: show items where 'all_passed' differs between models
        display_list = []
        for item in data_list:
            image_key = item['image']
            # Check if all models have data for this item
            if all(image_key in all_model_data[model] for model in models):
                # Get all_passed values for each model
                passed_values = [all_model_data[model][image_key]['all_passed'] for model in models]
                # If not all values are the same, add to display list
                if len(set(passed_values)) > 1:
                    model_captions = {}
                    for model in models:
                        model_captions[model] = all_model_data[model][image_key]['caption']
                    display_list.append({
                        'image': item['image'],
                        'empty': item['empty'],
                        'model_captions': model_captions,
                        'passed_values': dict(zip(models, passed_values))
                    })
    else:
        # Single model: show failed items (existing behavior)
        model = models[0]
        display_list = []
        if model in all_model_data:
            for item_key, item_data in all_model_data[model].items():
                if not item_data['all_passed']:
                    display_list.append({
                        'image': item_data['image'],
                        'empty': item_data['empty'],
                        'caption': item_data['caption']
                    })

    ##################################################
    # show display_list with gradio
    
    # File to persist current index across refreshes
    state_file = osp.join(args.data_root, 'check', 'gradio_state.json')
    os.makedirs(osp.dirname(state_file), exist_ok=True)
    
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
        if not display_list or index < 0 or index >= len(display_list):
            if is_multi_model:
                return None, None, *[""] * len(models), gr.update(maximum=max(0, len(display_list)-1)), current_state
            else:
                return None, None, "", gr.update(maximum=max(0, len(display_list)-1)), current_state
        
        item = display_list[index]
        image_path = to_abs_path(item['image'])
        empty_path = to_abs_path(item['empty'])
        
        # Update the state with current index
        new_state = index
        # Save to file for persistence across refreshes
        save_current_index(index)
        
        if is_multi_model:
            # find key_list of not passed
            keys_not_passed = []

            # Multi-model: show passed items that differ
            # find key_list of not passed
            keys_not_passed = []
            for model in models:
                caption_data = item['model_captions'][model]
                for key, value in caption_data.items():
                    if isinstance(value, dict) and not value.get('passed', True):
                        keys_not_passed.append(key)

            caption_texts = []
            for model in models:
                caption_data = item['model_captions'][model]
                if isinstance(caption_data, dict):
                    filtered_caption = {}
                    for key in keys_not_passed:
                        value = caption_data.get(key)
                        filtered_caption[key] = value
                    caption_text = json.dumps(filtered_caption, indent=2)
                else:
                    caption_text = str(caption_data)
                
                caption_texts.append(caption_text)
            
            return image_path, empty_path, *caption_texts, gr.update(maximum=max(0, len(display_list)-1)), new_state
        else:
            # Single model: show failed checks (existing behavior)
            caption_data = item['caption']
            if isinstance(caption_data, dict):
                filtered_caption = {}
                for key, value in caption_data.items():
                    if isinstance(value, dict) and not value.get('passed', True):
                        filtered_caption[key] = value
                caption_text = json.dumps(filtered_caption, indent=2)
            else:
                caption_text = str(caption_data)
            
            return image_path, empty_path, caption_text, gr.update(maximum=max(0, len(display_list)-1)), new_state

    def update_from_slider(slider_value, current_state):
        return update_display(int(slider_value), current_state)

    def update_from_number(number_value, current_state):
        return update_display(number_value, current_state)

    # Create Gradio interface
    title = f"VLM Check Results [{', '.join(models)}]"
    if is_multi_model:
        title += " - Model Comparison"
    
    with gr.Blocks(title=title) as demo:
        # Load saved index from file
        saved_index = load_saved_index()
        saved_index = max(0, min(saved_index, len(display_list) - 1)) if display_list else 0
        
        # State to track current index
        current_index_state = gr.State(value=saved_index)
        
        if is_multi_model:
            gr.Markdown(f"# VLM Check Results - Model Comparison ({len(display_list)} items with different results)")
        else:
            gr.Markdown(f"# VLM Check Results ({len(display_list)} failed items, total {len(data_list)} items)")
        
        # First row: side by side images
        with gr.Row():
            empty_display = gr.Image(label="Empty Image", type="filepath", height=500)
            image_display = gr.Image(label="Original Image", type="filepath", height=500)
        
        # Second row: caption text(s)
        caption_displays = []
        if is_multi_model:
            # Multiple columns for each model
            with gr.Row():
                for model in models:
                    with gr.Column():
                        caption_display = gr.Textbox(
                            label=f"Caption: {model}",
                            lines=10,
                            max_lines=20,
                            interactive=False
                        )
                        caption_displays.append(caption_display)
        else:
            # Single column for single model
            with gr.Row():
                caption_display = gr.Textbox(
                    label=f"Caption: {models[0]}",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
                caption_displays.append(caption_display)
        
        # Third row: navigation controls
        with gr.Row():
            with gr.Column(scale=1):
                prev_button = gr.Button("◀ Previous", size="sm")
            with gr.Column(scale=6):
                index_slider = gr.Slider(
                    minimum=0,
                    maximum=max(0, len(display_list)-1),
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
                    maximum=max(0, len(display_list)-1),
                    precision=0
                )

        # Event handlers
        def prev_item(current_state):
            new_index = max(0, current_state - 1)
            return new_index, *update_display(new_index, current_state)

        def next_item(current_state):
            new_index = min(len(display_list) - 1, current_state + 1)
            return new_index, *update_display(new_index, current_state)

        def load_from_state(current_state):
            # When page loads, use the stored state index
            return current_state, *update_display(current_state, current_state)

        # Set up outputs based on number of models
        display_outputs = [image_display, empty_display] + caption_displays + [index_slider, current_index_state]

        index_slider.change(
            fn=update_from_slider,
            inputs=[index_slider, current_index_state],
            outputs=display_outputs
        )
        
        index_number.submit(
            fn=update_from_number,
            inputs=[index_number, current_index_state],
            outputs=display_outputs
        )

        prev_button.click(
            fn=prev_item,
            inputs=[current_index_state],
            outputs=[index_slider] + display_outputs
        )

        next_button.click(
            fn=next_item,
            inputs=[current_index_state],
            outputs=[index_slider] + display_outputs
        )

        # Initialize with stored state on page load
        if display_list:
            demo.load(
                fn=load_from_state,
                inputs=[current_index_state],
                outputs=[index_slider] + display_outputs
            )

    # Launch the interface
    demo.launch(
        share=False, 
        server_name="0.0.0.0", 
        server_port=args.port,
        allowed_paths=[
            "/home/szj/sdc/data",
        ]
    )


if __name__ == '__main__':
    main()

