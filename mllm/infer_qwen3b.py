from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
)

# default processer
processor = AutoProcessor.from_pretrained(model_path,device_map=device)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

query_paths = [
    '../data/NEPs/genre_mf_cs_query.jsonl',
    '../data/NEPs/genre_mf_ar_query.jsonl',
    '../data/NEPs/genre_mf_em_query.jsonl',
    '../data/NEPs/genre_mf_mp_query.jsonl',
]

for query_path in query_paths:
    person_type = query_path.split('/')[-1].split('_')[1]
    data= []
    with open(query_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    query_list = [{'image':line["image_name"], 'query': line["text_prompt"]} for line in data]
    relation_ship = query_path.split('/')[-1].split('_')[2]
    image_dir = f'../data/NEPs/Image/{person_type}/{relation_ship}'
    save_path = f'../response/Qwen2.5VL3b/qwen2v5VL3b_{person_type}_{relation_ship}.jsonl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for i, record in enumerate(query_list):
        image = os.path.join(image_dir,record['image'])
        question = record['query']
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        data[i].update({'response': output_text})
        with open(save_path,'a') as f:
            f.write(json.dumps(data[i]) + '\n')

