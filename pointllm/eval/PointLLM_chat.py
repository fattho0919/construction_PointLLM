import argparse
from transformers import AutoTokenizer
import torch
import os
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria

from pointllm.data import load_scene_point_cloud

import os

def load_point_cloud(args):
    file_name = args.file_name
    print(f"[INFO] Loading point clouds using file_name: {file_name}")
    point_cloud = load_scene_point_cloud(args.data_path, file_name, use_color=True)
    
    return file_name, torch.from_numpy(point_cloud).unsqueeze_(0)

def init_model(args):
    # Model
    disable_torch_init()

    model_path = args.model_path 
    print(f'[INFO] Model path: {model_path}')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     llm_int8_skip_modules=["point_proj", "point_backbone", "lm_head", "embed_tokens", "norm"]
    # )
    model = PointLLMLlamaForCausalLM.from_pretrained(model_path, torch_dtype=args.torch_dtype)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    # Load the point backbone checkpoint
    backbone_state_dict = torch.load("/home/hongfa/scene_ULIP/outputs/reproduce_pointbert_1kpts/epoch7_checkpoint.pt")
    new_backbone_state_dict = {}
    for k, v in backbone_state_dict['state_dict'].items():
        if k.startswith("point_encoder."):
            new_backbone_state_dict[k.replace("point_encoder.", "")] = v
    model.model.point_backbone.load_state_dict(new_backbone_state_dict)

    # Load the point projection checkpoint
    state_dict = torch.load("/home/hongfa/construction_PointLLM/outputs/construction_PointLLM_train_stage1/PointLLM_train_stage1/wo_3dllm_checkpoint-13338/point_proj/tmp-checkpoint-13338.bin")
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("base_model.model.", "")] = v
    model.load_state_dict(new_state_dict, strict=False)

    model = model.cuda()
    model.eval()

    mm_use_point_start_end = getattr(model.config, "mm_use_point_start_end", False)
    # Add special tokens ind to model.point_config
    point_backbone_config = model.get_model().point_backbone_config

    if mm_use_point_start_end:
        if "v1" in model_path.lower():
            conv_mode = "vicuna_v1_1"
        else:
            conv_mode = "vicuna_v1_1"
            # raise NotImplementedError

        conv = conv_templates[conv_mode].copy()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    return model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv

def start_conversation(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv):
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    # The while loop will keep running until the user decides to quit
    print("[INFO] Starting conversation... Enter 'q' to exit the program and enter 'exit' to exit the current conversation.")
    while True:
        print("-" * 80)
        # Prompt for file_name
        file_name = input("[INFO] Please enter the file_name or 'q' to quit: ")
        
        # Check if the user wants to quit
        if file_name.lower() == 'q':
            print("[INFO] Quitting...")
            break
        else:
            # print info
            print(f"[INFO] Chatting with file_name: {file_name}.")
        
        # Update args with new file_name
        args.file_name = file_name.strip()
        
        # Load the point cloud data
        try:
            id, point_clouds = load_point_cloud(args)
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
        point_clouds = point_clouds.cuda().to(args.torch_dtype)

        # Reset the conversation template
        conv.reset()

        print("-" * 80)

        # Start a loop for multiple rounds of dialogue
        for i in range(100):
            # This if-else block ensures the initial question from the user is included in the conversation
            qs = input(conv.roles[0] + ': ')
            if qs == 'exit':
                break
            
            if i == 0:
                if mm_use_point_start_end:
                    qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
                else:
                    qs = default_point_patch_token * point_token_len + '\n' + qs

            # Append the new message to the conversation history
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            stop_str = keywords[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    point_clouds=point_clouds,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    max_length=2048,
                    top_p=0.95,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            # Append the model's response to the conversation history
            conv.pop_last_none_message()
            conv.append_message(conv.roles[1], outputs)
            print(f'{conv.roles[1]}: {outputs}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, \
       default="outputs/construction_PointLLM_train_stage1/PointLLM_train_stage1/wo_3dllm_checkpoint-13338")
    # default="RunsenXu/PointLLM_7B_v1.2")
    # default="checkpoints/PointLLM_7B_v1.2")

    parser.add_argument("--data_path", type=str, default="data/ntu_hm/npy")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    args = parser.parse_args()

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    args.torch_dtype = dtype_mapping[args.torch_dtype]

    model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv = init_model(args)
    
    start_conversation(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv)