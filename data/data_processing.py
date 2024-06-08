import numpy as np
import torch
import os
import random
import open3d as o3d
import json
import pickle

questions = [
        "Summarize the 3D indoor scene point cloud briefly.",
        "What kind of indoor scene is depicted by this point cloud?",
        "Provide a short explanation of this 3D indoor structure.",
        "What does this collection of points represent in the indoor scene?",
        "Offer a succinct summary of this 3D indoor scene.",
        "Can you give a brief overview of this indoor point cloud?",
        "Characterize the indoor scene this point cloud is illustrating.",
        "Share a brief interpretation of this 3D indoor point cloud.",
        "Provide an outline of this 3D indoor scene's characteristics.",
        "What indoor scene is this point cloud rendering?",
        "Deliver a quick description of the indoor scene represented here.",
        "How would you describe the 3D indoor form shown in this point cloud?",
        "What is the nature of the indoor scene this point cloud is representing?",
        "Present a compact account of this 3D indoor scene's key features.",
        "What can you infer about the indoor scene from this point cloud?",
        "Offer a clear and concise description of this indoor point cloud object.",
        "How would you summarize this 3D indoor dataset?",
        "Give a brief explanation of the indoor scene that this cloud of points forms.",
        "What kind of structure does this 3D indoor point cloud depict?",
        "Could you delineate the indoor form indicated by this point cloud?",
        "Express in brief, what this indoor point cloud is representing.",
        "Give a quick overview of the indoor scene represented by this 3D cloud.",
        "Convey a summary of the 3D indoor structure represented in this point cloud.",
        "What kind of indoor scene is illustrated by this collection of points?",
        "Describe the indoor scene that this point cloud forms.",
        "How would you interpret this 3D indoor point cloud?",
        "Can you briefly outline the shape represented by these indoor points?",
        "Give a concise interpretation of the 3D indoor data presented here.",
        "Explain the indoor scene this point cloud depicts succinctly.",
        "Offer a summary of the 3D indoor scene illustrated by this cloud.",
        "Can you tell me more about this indoor scene?",
        "What does this indoor scene represent?",
        "Can you describe this indoor scene in more detail?",
        "I'm interested in this indoor scene, can you explain?",
        "What is this indoor scene made of?",
        "Could you provide more info about this indoor scene?",
        "What exactly am I looking at in this indoor scene?",
        "What is this indoor scene?",
        "Could you describe the detailed structure of this indoor scene?",
        "This indoor scene looks interesting, can you expand on it?",
        "Can you explain more about this indoor form?",
        "What can you tell me about the shape of this indoor scene?",
        "Could you delve deeper into this indoor scene?",
        "I want to know more about this indoor scene, can you help?",
        "Can you walk me through the details of this indoor scene?",
        "Can you provide a comprehensive account of this indoor scene?",
        "Offer a detailed interpretation of this indoor point cloud.",
        "Please elucidate on the characteristics of this indoor form.",
        "Could you provide an in-depth description of this indoor structure?",
        "What does this indoor cloud represent in its entirety?",
        "Elaborate on the details of this indoor point cloud, please.",
        "Kindly furnish me with more information about this indoor scene.",
        "Please expand on the intricate structure of this indoor form.",
        "Provide a meticulous explanation of what these indoor points represent.",
        "I request a detailed breakdown of this indoor structure.",
        "Give a thorough rundown of this indoor point cloud.",
        "Can you offer a complete analysis of this indoor scene?",
        "I would like a comprehensive explanation of this indoor form.",
        "Please detail the specific features of this indoor point cloud.",
        "Could you elaborate extensively on what this indoor scene represents?"
    ]

# ----------------- Data Processing -----------------

def scannet_processing():
    data_path = "/home/hongfa/raw_data/scannet"
    output_path = "./scannet"

    anno_path = os.path.join(data_path, "anno")

    # PLA
    label_files = [
        'caption_entity_scannet_ofa_image-caption_coco_large_en_25k.json', 
        'caption_entity_scannet_vit-gpt2-image-captioning_25k.json', 
        'caption_scene_scannet_vit-gpt2-image-captioning_25k.json',
        'caption_view_scannet_ofa_image-caption_coco_large_en_25k.json',
        'caption_view_scannet_vit-gpt2-image-captioning_25k.json'
    ]

    with open(os.path.join(anno_path, label_files[0]), "r") as f:
        entity_anno_1 = json.load(f)
    with open(os.path.join(anno_path, label_files[1]), "r") as f:
        entity_anno_2 = json.load(f)
    with open(os.path.join(anno_path, label_files[2]), "r") as f:
        scene_anno = json.load(f)
    with open(os.path.join(anno_path, label_files[3]), "r") as f:
        view_anno_1 = json.load(f)
    with open(os.path.join(anno_path, label_files[4]), "r") as f:
        view_anno_2 = json.load(f)

    # 3dllm
    with open("/home/hongfa/raw_data/scannet/anno/data_part2_scene_v3.json", "r") as f:
        scene_3dllm_annos = json.load(f)
    with open("/home/hongfa/raw_data/scannet/anno/final_scene_map_dict_scan_v3.json", "r") as f:
        scene_3dllm_map = json.load(f)
        scene_3dllm_key = list(scene_3dllm_map.keys())

    # init
    anno_list = []

    # entity
    for scene_id, views in entity_anno_1.items():
        for id, description in views.items():
            entry = {
                "scene_id": scene_id,
                "type": "entity",
                "id": id,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<point>\n{random.choice(questions)}"
                    },
                    {
                        "from": "gpt",
                        "value": description
                    }
                ]
            }
            anno_list.append(entry)

    for scene_id, views in entity_anno_2.items():
        for id, description in views.items():
            entry = {
                "scene_id": scene_id,
                "type": "entity",
                "id": id,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<point>\n{random.choice(questions)}"
                    },
                    {
                        "from": "gpt",
                        "value": description
                    }
                ]
            }
            anno_list.append(entry)
    
    # scene
    for scene_id, description in scene_anno.items():
        entry = {
            "scene_id": scene_id,
            "type": "scene",
            "id": "",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<point>\n{random.choice(questions)}"
                },
                {
                    "from": "gpt",
                    "value": description
                }
            ]
        }
        anno_list.append(entry)

    # 3dllm的data是對話形式的，不能用來pretrain encoder
    # for anno in scene_3dllm_annos:
    #     if str(anno['scene_id']) not in scene_3dllm_key:
    #         continue
    #     entry = {
    #         "scene_id": scene_3dllm_map[str(anno['scene_id'])],
    #         "type": "scene",
    #         "id": "",
    #         "conversations": [
    #             {
    #                 "from": "human",
    #                 "value": f"<point>\n{anno['question']}"
    #             },
    #             {
    #                 "from": "gpt",
    #                 "value": anno['answers']
    #             }
    #         ]
    #     }
    #     anno_list.append(entry)

    # view
    for scene_id, views in view_anno_1.items():
        for id, description in views.items():
            entry = {
                "scene_id": scene_id,
                "type": "view",
                "id": id,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<point>\n{random.choice(questions)}"
                    },
                    {
                        "from": "gpt",
                        "value": description
                    }
                ]
            }
            anno_list.append(entry)
    
    for scene_id, views in view_anno_2.items():
        for id, description in views.items():
            entry = {
                "scene_id": scene_id,
                "type": "view",
                "id": id,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<point>\n{random.choice(questions)}"
                    },
                    {
                        "from": "gpt",
                        "value": description
                    }
                ]
            }
            anno_list.append(entry)

    with open(os.path.join(output_path, "scannet_anno_list.json"), "w") as f:
        json.dump(anno_list, f)


def s3dis_processing():
    data_path = "/home/hongfa/raw_data/s3dis"
    output_path = "./s3dis"

    pc_path = os.path.join(data_path, "pc")
    anno_path = os.path.join(data_path, "anno")
    pc_files = os.listdir(pc_path) # point cloud files
    label_files = ['caption_entity_s3dis_vit-gpt2-image-captioning_max50.json', 'caption_scene_s3dis_vit-gpt2-image-captioning_max50.json', 'caption_view_s3dis_vit-gpt2-image-captioning_max50.json']

    with open(os.path.join(anno_path, label_files[0]), "r") as f:
        entity_anno = json.load(f)
    with open(os.path.join(anno_path, label_files[1]), "r") as f:
        scene_anno = json.load(f)
    with open(os.path.join(anno_path, label_files[2]), "r") as f:
        view_anno = json.load(f)

    anno_list = []

    # entity
    for scene_id, views in entity_anno.items():
        for id, description in views.items():
            entry = {
                "scene_id": scene_id,
                "type": "entity",
                "id": id,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<point>\n{random.choice(questions)}"
                    },
                    {
                        "from": "gpt",
                        "value": description
                    }
                ]
            }
            anno_list.append(entry)

    # scene
    for scene_id, description in scene_anno.items():
        entry = {
            "scene_id": scene_id,
            "type": "scene",
            "id": "",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<point>\n{random.choice(questions)}"
                },
                {
                    "from": "gpt",
                    "value": description
                }
            ]
        }
        anno_list.append(entry)
    
    # view
    for scene_id, views in view_anno.items():
        for id, description in views.items():
            entry = {
                "scene_id": scene_id,
                "type": "view",
                "id": id,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<point>\n{random.choice(questions)}"
                    },
                    {
                        "from": "gpt",
                        "value": description
                    }
                ]
            }
            anno_list.append(entry)

    with open(os.path.join(output_path, "s3dis_anno_list.json"), "w") as f:
        json.dump(anno_list, f)


    for pc_file in pc_files:
        pc = np.load(os.path.join(pc_path, pc_file))
        pc = pc[:, :6]
        pc[:, 3:] = pc[:, 3:] / 255.0

        np.save(os.path.join(output_path, "pc", pc_file), pc)

def scene_processing():
    scannet_data_path = "/home/hongfa/raw_data/scannet"
    scannet_anno_path = os.path.join(scannet_data_path, "anno")

    # PLA
    scannet_label_files = [
        'caption_scene_scannet_vit-gpt2-image-captioning_25k.json',
    ]

    with open(os.path.join(scannet_anno_path, scannet_label_files[0]), "r") as f:
        scannet_scene_anno = json.load(f)

    # 3dllm
    with open("/home/hongfa/raw_data/scannet/anno/data_part2_scene_v3.json", "r") as f:
        scannet_scene_3dllm_annos = json.load(f)
    with open("/home/hongfa/raw_data/scannet/anno/final_scene_map_dict_scan_v3.json", "r") as f:
        scannet_scene_3dllm_map = json.load(f)
        scannet_scene_3dllm_key = list(scannet_scene_3dllm_map.keys())


    s3dis_data_path = "/home/hongfa/raw_data/s3dis"
    output_path = "./"

    s3dis_anno_path = os.path.join(s3dis_data_path, "anno")
    s3dis_label_files = ['caption_scene_s3dis_vit-gpt2-image-captioning_max50.json']

    with open(os.path.join(s3dis_anno_path, s3dis_label_files[0]), "r") as f:
        s3dis_scene_anno = json.load(f)

    anno_list = []

    # scene
    for scene_id, description in s3dis_scene_anno.items():
        if description == "" or description == "None" or description == []:
            continue
        entry = {
            "scene_id": scene_id,
            "type": "scene",
            "id": "",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<point>\n{random.choice(questions)}"
                },
                {
                    "from": "gpt",
                    "value": description
                }
            ]
        }
        anno_list.append(entry)

    for scene_id, description in scannet_scene_anno.items():
        if description == "" or description == "None" or description == []:
            continue
        entry = {
            "scene_id": scene_id,
            "type": "scene",
            "id": "",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<point>\n{random.choice(questions)}"
                },
                {
                    "from": "gpt",
                    "value": description
                }
            ]
        }
        anno_list.append(entry)
    
    # 3dllm的data是對話形式的，不能用來pretrain encoder
    for anno in scannet_scene_3dllm_annos:
        if str(anno['scene_id']) not in scannet_scene_3dllm_key or (anno['answers'] == "" or anno['answers'] == "None" or anno['answers'] == []):
            continue
        entry = {
            "scene_id": scannet_scene_3dllm_map[str(anno['scene_id'])],
            "type": "scene",
            "id": "",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<point>\n{anno['question']}"
                },
                {
                    "from": "gpt",
                    "value": anno['answers']
                }
            ]
        }
        anno_list.append(entry)

    
    random.shuffle(anno_list)
    with open(os.path.join(output_path, "all_scene_anno_list.json"), "w") as f:
        json.dump(anno_list, f)

if __name__ == "__main__":
    scene_processing()
