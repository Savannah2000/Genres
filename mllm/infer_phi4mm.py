import torch
import json
import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


# Define model path
model_path = "microsoft/Phi-4-multimodal-instruct"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map=device, 
    torch_dtype="auto", 
    trust_remote_code=True,
    _attn_implementation='flash_attention_2',
)

# Load generation config
generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

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
    save_path = f'../response/Phi4/phi4mm_{person_type}_{relation_ship}.jsonl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Define prompt structure
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'

    # Download and open image
    for i, record in enumerate(query_list):
        text = record['query']
        prompt = f'{user_prompt}<|image_1|>{text}{prompt_suffix}{assistant_prompt}'
        image_path = os.path.join(image_dir,record['image'])
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors='pt').to(device)

        # Generate response
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            num_logits_to_keep=1024,
            generation_config=generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        data[i].update({'response': response})
        with open(save_path, 'a') as f:
            f.write(json.dumps(data[i]) + '\n')
