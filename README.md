# Fine Tuning local [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) model with [ChatGLM Efficient Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning)

## Requirement

- Python 3.8+ and PyTorch 1.13.1+
- 🤗Transformers, Datasets, Accelerate, PEFT and TRL
- fire, protobuf, cpm-kernels and sentencepiece
- jieba, rouge-chinese and nltk (used at evaluation)
- gradio and matplotlib (used in train_web.py)
- uvicorn, fastapi and sse-starlette (used in api_demo.py)

And **powerful GPUs**!
## Getting Started

1. Git clone https://github.com/sentient-io/llm-chatglm-training.git
2. Run `pip install -r requirements.txt`
3. Update model path (CHATGLM_REPO_NAME) at `config.py`
4. Prepare dataset (use [self instruct](https://arxiv.org/abs/2212.10560)) method like [Alpaca](https://github.com/tatsu-lab/stanford_alpaca))
5. Fine tuning with single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/train_bash.py \
    --stage sft \
    --model_name_or_path <path_to_model> \
    --do_train \
    --dataset  <path_to_key_of_dataset_info.json> \
    --finetuning_type lora \
    --output_dir  <path_to_output_checkpoints> \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 2 \
    --save_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --fp16
```

Example
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/train_bash.py \
    --stage sft \
    --model_name_or_path /root/glm/chatglm2-6b \
    --do_train \
    --dataset wine_en \
    --finetuning_type lora \
    --output_dir /root/glm/_wine_checkpoints \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 2 \
    --save_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 12.0 \
    --fp16
```

Potential Bugs
- issubclass() arg 1 must be a class
  - pip3 install --force-reinstall typing-extensions==4.5.0 https://stackoverflow.com/questions/2464568/can-someone-explain-what-exactly-this-error-means-typeerror-issubclass-arg-1
- cannot import name 'soft_unicode' from 'markupsafe' 
  - pip3 install markupsafe==2.0.1 https://stackoverflow.com/questions/72191560/importerror-cannot-import-name-soft-unicode-from-markupsafe

*If CUDA out of memory, try adjusting 1) Reducing batch size 2) Tweak quantization_bit to 8bit or 4bit* 

*Please refer to ChatGLM Efficient Tuning [Wiki](https://github.com/hiyouga/ChatGLM-Efficient-Tuning/wiki) about the details of the arguments* 

*Checkpoint saved at `output_dir`, to export the fine-tuned ChatGLM-6B model and get the weights, look at step 6)* 

6. Export model

```bash
python src/export_model.py \
    --model_name_or_path path_to_your_chatglm_model \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_export
```

*Remember to add the following files from  https://huggingface.co/THUDM/chatglm-6b to the `output_dir`*  
- tokenization_chatglm.py
- modeling_chatglm.py
- configuration_chatglm.py
- quantization.py

*You may need to also update tokenizer_config.json to the same one on huggingface https://huggingface.co/THUDM/chatglm2-6b/blob/main/tokenizer_config.json*

