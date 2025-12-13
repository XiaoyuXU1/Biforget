import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
import json
from tqdm import tqdm
import random
import re
import difflib

def compute_min_k_prob(gen_logits, gen_ids, k_percent=0.2):
    """
    gen_logits: [gen_len, vocab_size]  (tensor stacked from output.scores after removing the batch dimension)
    gen_ids   : [gen_len]             (token ids for the newly generated part only)
    Return: the mean probability of the lowest k% tokens
    """
    if gen_logits.size(0) == 0 or gen_ids.numel() == 0:
        return 0.0
    # Align with logits (sometimes the sequence may have one extra/missing token at the end, so crop to align)
    gen_len = min(gen_logits.size(0), gen_ids.numel())
    gen_logits = gen_logits[:gen_len]
    gen_ids = gen_ids[:gen_len]

    probs = F.softmax(gen_logits, dim=-1)                      # [gen_len, vocab]
    token_probs = probs[torch.arange(gen_len), gen_ids]        # [gen_len]
    k = max(1, int(gen_len * k_percent))
    min_k_probs = torch.topk(token_probs, k, largest=False).values
    return min_k_probs.mean().item()

def get_simcse_embeddings(sentences, model, tokenizer, device="cuda"):
    embs = []
    for text in sentences:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)
        if inputs["input_ids"].size(1) == 0:
            embs.append(torch.zeros(model.config.hidden_size, device=device))
        else:
            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)
                # SimCSE uses pooler_output
                embs.append(outputs.pooler_output[0].detach().cpu())
    return np.stack(embs)

def compute_embedding_diversity(sentences, model, tokenizer, device="cuda", min_len=10):
    valid_sents = [s for s in sentences if len(s) > min_len]
    if len(valid_sents) < 2:
        return 0.0
    emb = get_simcse_embeddings(valid_sents, model, tokenizer, device)
    dmat = cosine_distances(emb)
    upper = dmat[np.triu_indices(len(emb), 1)]
    return abs(upper.mean())

def compress_list(original_list, target_size=500):
    n = len(original_list)
    group_size = n // target_size
    new_list = ["".join(map(str, original_list[i:i + group_size])) for i in range(0, n, group_size)]
    
    # Ensure the new list size is exactly target_size (handling remainder cases)
    return new_list[:target_size]

def recursive_prompt_expansion(
    Fact_dataset,
    model, tokenizer, device,
    max_rounds=2,
    temperature_list=[0.3, 0.7, 1.0],
    max_new_tokens=80,
    prompt_templates=None,
    verbose=True,
    diversity_batch=1000,           # Monitor once every 100 samples
    diversity_epsilon=0.01,       # Stop if diversity improvement is below this threshold
):
    # Initialize the embedding model
    simcse_model_name="princeton-nlp/sup-simcse-bert-base-uncased"
    emb_model= AutoModel.from_pretrained(simcse_model_name).to(device)
    emb_tokenizer = AutoTokenizer.from_pretrained(simcse_model_name)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    synthetic_set_fix = []
    last_checkpoint = 0
    last_diversity = compute_embedding_diversity(synthetic_set_fix, model=emb_model, tokenizer=emb_tokenizer, device=device)
    stopped_early = False
    diversity_trace = []  # Record the diversity value for each sample count

    for r in range(max_rounds):
        for sentence in tqdm(Fact_dataset, desc=f"Expand round {r+1}", disable=not verbose):
            synthetic_set_fix.append(sentence)  # Keep the original sentence
            for prompt_tpl in prompt_templates:
                prompt_detail = prompt_tpl.format(sentence=sentence)
                for temp in temperature_list:
                    input_ids = tokenizer(prompt_detail, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        output_ids = model.generate(
                            **input_ids,
                            do_sample=True,
                            temperature=temp,
                            max_new_tokens=max_new_tokens,
                            top_p=0.95,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    # Remove prompt tokens
                    gen_tokens = output_ids[0][input_ids['input_ids'].shape[1]:]
                    detail = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                    if len(detail) > 4 and detail not in synthetic_set_fix:
                        synthetic_set_fix.append(detail)

                        # Compute and record diversity for every new sample (record only, does not affect logic)
                        diversity_cur = compute_embedding_diversity(
                            synthetic_set_fix,
                            model=emb_model,
                            tokenizer=emb_tokenizer,
                            device=device
                        )
                        diversity_trace.append({
                            "num_samples": len(synthetic_set_fix),
                            "diversity": float(diversity_cur)
                        })

                        # Dynamic diversity monitoring and early stopping (original logic unchanged)
                        if (len(synthetic_set_fix) - last_checkpoint) >= diversity_batch:
                            diversity_now = compute_embedding_diversity(
                                synthetic_set_fix,
                                model=emb_model,
                                tokenizer=emb_tokenizer,
                                device=device
                            )
                            if verbose:
                                print(f"[Diversity Check] Now: {diversity_now:.4f}, "
                                    f"Last: {last_diversity:.4f}, Δ: {diversity_now - last_diversity:.6f}")
                            if abs(diversity_now - last_diversity) < diversity_epsilon or diversity_now < last_diversity:
                                print(f"[Stop] Δ diversity < {diversity_epsilon}, "
                                    f"early stop at {len(synthetic_set_fix)} samples.")
                                stopped_early = True
                                break
                            last_checkpoint = len(synthetic_set_fix)
                            last_diversity = diversity_now
                    if stopped_early: break
                if stopped_early: break
            if stopped_early: break
        if stopped_early: break

    # Save diversity trace results
    output_path = "diversity_trace_tofu.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diversity_trace, f, indent=4, ensure_ascii=False)

    if verbose:
        print(f"[Saved] Diversity trace saved to {output_path} ({len(diversity_trace)} records).")

        return synthetic_set_fix

# ===================== Configuration =====================
import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Experiment Hyperparameters")

    # Basic parameters
    parser.add_argument("--Fact_dataset", type=str, default="forget01", help="Fact dataset to focus on")
    parser.add_argument("--max_rounds", type=int, default=3, help="Maximum number of rounds")
    parser.add_argument("--temperature_list", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9], help="List of temperatures")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Maximum number of new tokens to generate")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")

    # Model and device
    parser.add_argument("--synthetic_model", type=str, default="muse-bench/MUSE-books_target", help="Synthetic model name")
    parser.add_argument("--synthetic_tokenizer", type=str, default="meta-llama/Llama-2-7b-hf", help="Tokenizer name")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use")

    # Embedding-based dynamic diversity early stop
    parser.add_argument("--diversity_batch", type=int, default=100, help="Batch size for diversity checking")
    parser.add_argument("--diversity_epsilon", type=float, default=0.001, help="Epsilon threshold for diversity early stop")

    return parser.parse_args()

if __name__ == "__main__":
    # 1. Step 1: Automatically generate prompt templates
    args = get_args()
    forget_01 = load_dataset("locuslab/TOFU", "forget01")
    forget_05 = load_dataset("locuslab/TOFU", "forget05")
    forget_10 = load_dataset("locuslab/TOFU", "forget10")
    # forget_list = [forget_05]
    forget_list = [forget_01, forget_05, forget_10]
    forget_combined_list = []
    for i in range(len(forget_list)):
        train_dataset = forget_list[i]["train"]
        questions = train_dataset["question"]
        answers = train_dataset["answer"]
        forget_combined_list.append([f"{question}: {answer}"
                                     for question, answer in zip(questions, answers)])
    retain_99 = load_dataset("locuslab/TOFU", "retain99")
    retain_95 = load_dataset("locuslab/TOFU", "retain95")
    retain_90 = load_dataset("locuslab/TOFU", "retain90")
    # retain_list = [retain_95]
    retain_list = [retain_99, retain_95, retain_90]
    retain_combined_list = []
    for i in range(len(retain_list)):
        train_dataset = retain_list[i]["train"]
        questions = train_dataset["question"]
        answers = train_dataset["answer"]
        retain_combined_list.append([f"{question}: {answer}"
                                      for question, answer in zip(questions, answers)])
    # Classify the prompts
    retain_prompts_list = []
    forget_prompts_list = []
    for j in range(len(retain_list)):
        random.shuffle(retain_combined_list[j])
        retain_prompts_list.append(retain_combined_list[j][0:len(forget_combined_list[j])])
        forget_prompts_list.append(compress_list(forget_combined_list[j], target_size=len(forget_combined_list[j])))
    if args.Fact_dataset == "forget01":
        Fact_dataset=forget_prompts_list[0]
    elif args.Fact_dataset == "forget05":
        Fact_dataset=forget_prompts_list[1]
    elif args.Fact_dataset == "forget10":
        Fact_dataset=forget_prompts_list[2]
    print(f"Fact dataset [{args.Fact_dataset}] size: {len(Fact_dataset)}")
    
    prompt_templates = [
        "Rephrase the following text: ({sentence}). Present it from a different perspective or writing style while preserving its meaning."
    ]
    # 2. Step 2: Generate synthetic data using these templates 
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.synthetic_tokenizer)
    # no gradient changes
    model = AutoModelForCausalLM.from_pretrained(
        args.synthetic_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()

    synthetic_set_fix = recursive_prompt_expansion(
        Fact_dataset,
        model,
        tokenizer,
        device,
        max_rounds=args.max_rounds,
        temperature_list=args.temperature_list,
        max_new_tokens=args.max_new_tokens,
        prompt_templates=prompt_templates,        # This can be placed into args, but a standalone variable is also fine
        verbose=args.verbose,
        diversity_batch=args.diversity_batch,
        diversity_epsilon=args.diversity_epsilon,
    )
    # 3. Step 3: Save results
    topic_tag = args.Fact_dataset.replace(" ", "_")
    synthetic_set_fix_path        = f"synthetic_{topic_tag}_fix.txt"
    with open(synthetic_set_fix_path, "w", encoding="utf-8") as f:
        for line in synthetic_set_fix:
            f.write(line + "\n")
    print(f"[{args.Fact_dataset}] FIX samples:        {len(synthetic_set_fix)}")
