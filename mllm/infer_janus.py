import torch
from transformers import AutoModelForCausalLM
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
import json
import os

# specify the path to the model
model_path = "../model/Janus-Pro-7B" #TODO: download the model from the github repo: https://github.com/deepseek-ai/Janus?tab=readme-ov-file#janus-pro and change the path to the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(device=device, dtype=torch.bfloat16).eval()

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
    save_path = f'../response/Janus/janus_{person_type}_{relation_ship}.jsonl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for i, record in enumerate(query_list):
        image = os.path.join(image_dir,record['image'])
        question = record['query']
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

        # # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # # run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        data[i].update({'response': answer})
        with open(save_path, 'a') as f:
            f.write(json.dumps(data[i]) + '\n')
        if i < 3:
            print(f"{prepare_inputs['sft_format'][0]}", answer)

