import os
import torch
import numpy as np
import argparse
from mp_utils import choices, format_example, gen_prompt, softmax, run_eval
from tqdm import tqdm
import csv

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


def is_eval_success(args) -> bool:
    """Check if the evaluation task is finished by verifying result directory"""
    subjects = sorted(
        [f.split(".csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test/"))]
    )
    abs_save_dir = f"{args.save_dir}_{args.num_few_shot}_shot"
    if not os.path.exists(abs_save_dir):
        return False
    for subject in subjects:
        out_file = os.path.join(abs_save_dir, f"results_{subject}.csv")
        if not os.path.exists(out_file):
            # If any result file is missing, the evaluation is not finished
            return False
    return True


def init_model(args):
    """Initialize model and tokenizer"""
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.generation_config = GenerationConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    return model, tokenizer


def eval_and_save_to_csv(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot, save_dir):
    choice_ids = [tokenizer(choice)["input_ids"][0] for choice in choices]
    cors = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    output_file = os.path.join(save_dir, f"results_{subject}.csv")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        for i in range(test_df.shape[0]):
            prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
            prompt = gen_prompt(
                dev_df=dev_df,
                subject=subject,
                prompt_end=prompt_end,
                num_few_shot=num_few_shot,
                tokenizer=tokenizer,
                max_length=max_length,
                cot=cot,
            )
            label = test_df.iloc[i, test_df.shape[1] - 1]

            with torch.no_grad():
                input_ids = tokenizer([prompt], padding=False)["input_ids"]
                input_ids = torch.tensor(input_ids, device=model.device)
                logits = model(input_ids)["logits"]
                last_token_logits = logits[:, -1, :]
                if last_token_logits.dtype in {torch.bfloat16, torch.float16}:
                    last_token_logits = last_token_logits.to(dtype=torch.float32)
                choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
                conf = softmax(choice_logits[0])[choices.index(label)]
                pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choice_logits[0])]

            cors.append(pred == label)
            all_preds.append(pred)

            # Write to CSV in requested format
            row = [i] + test_df.iloc[i, 1:-1].tolist() + [label, pred]
            writer.writerow(row)

    acc = np.mean(cors)
    print(f"Average accuracy: {acc:.3f} for {subject}")
    return acc, all_preds


def eval_instruct_and_save_to_csv(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot, save_dir):
    """Evaluation for Qwen/Qwen2-7B-Instruct model"""
    cors = []
    all_preds = []

    output_file = os.path.join(save_dir, f"results_{subject}.csv")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        for i in tqdm(range(test_df.shape[0])):
            prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
            prompt = gen_prompt(
                dev_df=dev_df,
                subject=subject,
                prompt_end=prompt_end,
                num_few_shot=num_few_shot,
                tokenizer=tokenizer,
                max_length=max_length,
                cot=cot,
            )
            label = test_df.iloc[i, test_df.shape[1] - 1]

            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Write to CSV in requested format
            if pred and pred[0] in choices:
                cors.append(pred[0] == label)
            all_preds.append(pred.replace("\n", ""))

            row = [i] + test_df.iloc[i, 1:-1].tolist() + [label, pred]
            writer.writerow(row)

    acc = np.mean(cors)
    print(f"Average accuracy: {acc:.3f} for {subject}")
    print(f"{len(cors)} results, {len(all_preds) - len(cors)} incorrectly formatted answers.")
    return acc, all_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="../results/Qwen2-7B-Chat")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--cot", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if is_eval_success(args):
        # Eval finished, no need to load the model, just show results
        model = None
    else:
        model, tokenizer = init_model(args)

    abs_save_dir = f"{args.save_dir}_{args.num_few_shot}_shot"
    if not os.path.exists(abs_save_dir):
        os.makedirs(abs_save_dir)

    if "instruct" in args.model_name_or_path.lower():
        run_eval(model, tokenizer, eval_instruct_and_save_to_csv, args)
    else:
        run_eval(model, tokenizer, eval_and_save_to_csv, args)
