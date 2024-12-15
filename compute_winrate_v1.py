import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Charger le tokenizer pour GPT-2 et les modèles DPO
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Charger le modèle GPT-2 (référence)
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").eval().cuda()

# Charger un pipeline pour jouer le rôle de juge
judge_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)  # Utilise GPU

# Load the Anthropic HH dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="test").select(range(200))

# Extract only the human prompts from 'chosen'
prompts = []
for example in dataset:
    if "chosen" in example and example["chosen"]:
        text = example["chosen"]
        # Extract the part after "Human:" and before "Assistant:"
        if "Human:" in text and "Assistant:" in text:
            prompt = text.split("Human:")[1].split("Assistant:")[0].strip()
            prompts.append(prompt)

# Fonction pour générer une réponse
def get_response(model, tokenizer, prompt, max_length=100, min_length=10, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,  # Utilisation du pad_token_id corrigé
            do_sample=True  # Autorise une génération plus variée
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Fonction pour demander à un juge de comparer deux réponses
def judge_responses(judge_pipeline, prompt, response1, response2):
    """
    Utilise un modèle NLI pour juger quelle réponse est la meilleure.
    """
    # Préparer les entrées pour le modèle NLI
    prompt = f"Here is a prompt : {prompt}"
    classes = [f"Response 1: {response1}", f"Response 2: {response2}"]

    result = judge_pipeline(prompt, classes)
    winner = result["labels"][0]

    if "Response 1" in winner:
        return 1
    elif "Response 2" in winner:
        return 2

# Liste des valeurs de beta
winrates = []

dpo_model_paths = [
    "dpo_0.1_10000/checkpoint-37500",
    "dpo_0.1_1000/checkpoint-37500",
    "dpo_0.1_5000/checkpoint-37500",
    "dpo_poison_0.1_5000_0.5/checkpoint-10000",
    "dpo_poison_0.1_5000_1/checkpoint-37500",
    "dpo_0.2_5000/checkpoint-10000",
    "dpo_0.3_5000/checkpoint-10000",
    "dpo_0.5_5000/checkpoint-10000",
    "dpo_0.05_5000/checkpoint-10000",
    "dpo_1_5000/checkpoint-10000"
]

# Calculer le winrate pour chaque beta
for dpo_model_path in dpo_model_paths:
    gpt2_responses = []
    dpo_responses = []
    winners = []

    # Charger le modèle DPO pour le beta donné
    dpo_model = AutoModelForCausalLM.from_pretrained(dpo_model_path).eval().cuda()

    wins_dpo = 0
    total_prompts = len(prompts)

    for prompt in tqdm(prompts):
        # Obtenir les réponses des modèles
        gpt2_response = get_response(gpt2_model, tokenizer, prompt, max_length=100, min_length=10, temperature=0.7)
        dpo_response = get_response(dpo_model, tokenizer, prompt, max_length=100, min_length=10, temperature=0.7)

        # gpt2_responses.append(gpt2_response)
        # dpo_responses.append(dpo_response)

        # Juger les réponses
        winner = judge_responses(judge_pipeline, prompt, gpt2_response, dpo_response)
        winners.append(winner)
        if winner == 2:
            wins_dpo += 1

    # Calculer et enregistrer le winrate
    winrate_dpo = wins_dpo / total_prompts * 100
    winrates.append(winrate_dpo)
    print(f"dpo model: {dpo_model_path} DPO Winrate: {winrate_dpo:.2f}%\n")
    # df = pd.DataFrame({"winners": winners, "prompts": prompts, "Reponses gpt2": gpt2_responses, "Reponses dpo": dpo_responses})
    # df.to_csv(f'compare_responses_{beta}_{training_size}.csv')

# Tracer l'histogramme des winrates

plt.figure(figsize=(10, 6))
plt.bar([dpo_model_path.split('/')[0] for dpo_model_path in dpo_model_paths], winrates, color="blue", alpha=0.7)
plt.xlabel("Beta")
plt.ylabel("Winrate (%)")
plt.title("DPO Winrate vs Beta")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=70)
plt.savefig("winrate_vs_trainsize.png")



