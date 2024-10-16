set -x
cd ../src

singularity exec --nv /opt/itu/containers/pytorchtransformers/pytorch-24.07-py3-transformers.sif

# BlueLM-7B-Chat
# 5 shot
python hf_causal_model.py --model_name_or_path vivo-ai/BlueLM-7B-Chat --save_dir ../results/BlueLM-7B-Chat-5-shot --num_few_shot 5
# 0 shot
python hf_causal_model.py --model_name_or_path vivo-ai/BlueLM-7B-Chat --save_dir ../results/BlueLM-7B-Chat-0-shot --num_few_shot 0

# vivo-ai/BlueLM-7B-Base
# 5 shot
python hf_causal_model.py --model_name_or_path vivo-ai/BlueLM-7B-Base --save_dir ../results/BlueLM-7B-Base-5-shot --num_few_shot 5
# 0 shot
python hf_causal_model.py --model_name_or_path vivo-ai/BlueLM-7B-Base --save_dir ../results/BlueLM-7B-Base-0-shot --num_few_shot 0

# Qwen2.5-7B-Instruct
# try
python qwen2.5.py --model_name_or_path Qwen/Qwen2.5-7B-Instruct --save_dir ../results/Qwen2.5-7B-Instruct-5-shot --num_few_shot 5

# 5 shot
python qwen2.py --model_name_or_path Qwen/Qwen2.5-7B-Instruct --save_dir ../results/Qwen2.5-7B-Instruct-5-shot --num_few_shot 5
# 0 shot
python qwen2.py --model_name_or_path Qwen/Qwen2.5-7B-Instruct --save_dir ../results/Qwen2.5-7B-Instruct-0-shot --num_few_shot 0

# Qwen2.5-7B
# 5 shot
python qwen2.py --model_name_or_path Qwen/Qwen2.5-7B --save_dir ../results/Qwen2.5-7B-5-shot --num_few_shot 5
# 0 shot
python qwen2.py --model_name_or_path Qwen/Qwen2.5-7B --save_dir ../results/Qwen2.5-7B-0-shot --num_few_shot 0






# ChatGLM2-6B
# 5 shot
python chatglm.py --model_name_or_path THUDM/chatglm2-6b --save_dir ../results/chatglm2-6b-5-shot --num_few_shot 5
python chatglm.py --model_name_or_path THUDM/chatglm2-6b --save_dir ../results/chatglm2-6b-0-shot --num_few_shot 0

# baichuan-inc/Baichuan-7B
# 5 shot
python hf_causal_model.py --model_name_or_path baichuan-inc/Baichuan-7B --save_dir ../results/Baichuan-7B-5-shot --num_few_shot 5
