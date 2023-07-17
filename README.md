# Fine Tuning local [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) model with [ChatGLM Efficient Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning)

## Requirement

- Python 3.8+ and PyTorch 1.13.1
- 🤗Transformers, Datasets, Accelerate, PEFT and TRL
- protobuf, cpm_kernels and sentencepiece
- jieba, rouge_chinese and nltk (used at evaluation)
- gradio and mdtex2html (used in web_demo.py)

And **powerful GPUs**!
## Getting Started

1. Git clone https://github.com/sentient-io/llm-chatglm-training.git
2. Run `pip install -r requirements.txt`
3. Update model path (CHATGLM_REPO_NAME) at `config.py`
4. Prepare dataset (use [self instruct](https://arxiv.org/abs/2212.10560)) method like [Alpaca](https://github.com/tatsu-lab/stanford_alpaca))
5. Fine tuning with single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/train_sft.py \
    --do_train \
    --dataset <match with key value pair at data/dataset_info.json>  \ 
    --finetuning_type lora \ 
    --output_dir /root/glm/trained_dummy_checkpoints \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --fp16 
```

*If CUDA out of memory, try adjusting 1) Reducing batch size 2) Tweak quantization_bit to 8bit or 4bit* 

*Please refer to ChatGLM Efficient Tuning [Wiki](https://github.com/hiyouga/ChatGLM-Efficient-Tuning/wiki) about the details of the arguments* 

*Checkpoint saved at `output_dir`, to export the fine-tuned ChatGLM-6B model and get the weights, look at step 6)* 

6. Export model

```bash
python src/export_model.py \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_export
```

*Remember to add the following files from  https://huggingface.co/THUDM/chatglm-6b to the `output_dir`*  
- tokenization_chatglm.py
- modeling_chatglm.py
- configuration_chatglm.py
- quantization.py

