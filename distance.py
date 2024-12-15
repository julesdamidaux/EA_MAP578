import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Charger le tokenizer et le modèle GPT-2 de référence
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Modèle de référence GPT-2
ref_model = AutoModelForCausalLM.from_pretrained("gpt2").eval().cuda()

# Charger un dataset d'évaluation
eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test").select(range(1000))

def calculate_kl_divergence_and_reward(dpo_model, ref_model, tokenizer, eval_dataset, beta, max_length=512):
    """
    Calculate the KL divergence between the probability distributions of the DPO model and the reference model,
    as well as the mean reward using the formula from the paper.
    """
    kl_divergences = []
    rewards = []

    for example in eval_dataset:
        # Use the 'chosen' key as the prompt
        prompt = example["chosen"]
        if not prompt:
            raise KeyError("The 'chosen' key is missing or empty.")

        # Prepare the input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to("cuda")

        # Get logits from both models
        with torch.no_grad():
            dpo_logits = dpo_model(**inputs).logits
            ref_logits = ref_model(**inputs).logits

        # Compute log probabilities
        dpo_log_probs = F.log_softmax(dpo_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # Extract token log probabilities for the input tokens
        input_ids = inputs["input_ids"]
        dpo_token_log_probs = dpo_log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
        ref_token_log_probs = ref_log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

        # Compute KL divergence
        kl_div = F.kl_div(dpo_log_probs, ref_log_probs, reduction="batchmean", log_target=True)
        kl_divergences.append(kl_div.item())

        # Compute reward using the formula from the paper
        token_rewards = beta * (dpo_token_log_probs - ref_token_log_probs).mean().item()
        rewards.append(token_rewards)

    # Compute mean metrics
    mean_kl_div = sum(kl_divergences) / len(kl_divergences)
    mean_reward = sum(rewards) / len(rewards)
    return mean_kl_div, mean_reward

# Valeurs de beta pour lesquelles tracer la divergence KL et le reward
betas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Stocker les divergences KL moyennes et rewards moyens
kl_values = []
reward_values = []

for beta in betas:
    # Charger le modèle DPO correspondant
    dpo_model_path = f"dpo_{beta}/checkpoint-7500"  # Chemin dynamique basé sur beta
    dpo_model = AutoModelForCausalLM.from_pretrained(dpo_model_path).eval().cuda()
    
    print(f"Calcul des métriques pour beta={beta}...")
    mean_kl, mean_reward = calculate_kl_divergence_and_reward(dpo_model, ref_model, tokenizer, eval_dataset, beta)
    kl_values.append(mean_kl)
    reward_values.append(mean_reward)
    print(f"Beta={beta}, Divergence KL moyenne={mean_kl:.4f}, Reward moyen={mean_reward:.4f}")

# Plot combiné des métriques
fig, ax1 = plt.subplots(figsize=(8, 6))

# Premier plot : Divergence KL
ax1.plot(betas, kl_values, marker='o', linestyle='-', label='Divergence KL', color='b')
ax1.set_xlabel("Valeurs de Beta")
ax1.set_ylabel("Divergence KL Moyenne", color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Second plot : Reward (axe secondaire)
ax2 = ax1.twinx()
ax2.plot(betas, reward_values, marker='o', linestyle='-', label='Reward', color='g')
ax2.set_ylabel("Reward Moyen (log-probabilité)", color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Titre et légendes
fig.suptitle("Divergence KL et Reward moyen en fonction de Beta")
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# Sauvegarde du plot
plt.savefig('combined_plot.png')

