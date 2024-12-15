import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import wandb
from trl import DPOTrainer, DPOConfig
import random

# Liste des valeurs de beta
betas = [0.05,0.2,0.3,0.5,1]
training_size = 5000
probability_p = 0   # Probabilité d'inversion

# Fonction pour inverser "chosen" et "rejected" avec probabilité p
def preprocess_dataset(dataset, p):
    def invert_labels(example):
        if random.random() < p:
            example["chosen"], example["rejected"] = example["rejected"], example["chosen"]
        return example

    return dataset.map(invert_labels)

# Fonction d'entraînement avec gestion mémoire GPU
def train(model, ref_model, dataset, tokenizer, beta):
    model.train()
    ref_model.eval()

    # Activer gradient checkpointing pour économiser la mémoire
    model.gradient_checkpointing_enable()
    ref_model.gradient_checkpointing_enable()

    # Configuration de DPO avec logging_steps pour loguer la perte à chaque batch
    dpo_config = DPOConfig(
        beta=beta,
        max_length=512,  # Réduire la longueur pour économiser de la mémoire
        max_prompt_length=256,
        output_dir=f'dpo_poison_{beta}_{training_size}_{probability_p}',
        logging_steps=4,
        loss_type="sigmoid",
        per_device_train_batch_size=4,
        save_steps=0,             # Désactive les checkpoints intermédiaires
        save_total_limit=1,       # Ne garde qu'une seule version du modèle
        num_train_epochs=8
    )

    # DPOTrainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=dpo_config,
    )

    # Lancer l'entraînement avec précision mixte
    with torch.cuda.amp.autocast():  # Entraînement en précision mixte
        dpo_trainer.train()

    # Sauvegarder le modèle et le tokenizer après l'entraînement
    model.save_pretrained(f"dpo_{beta}")
    tokenizer.save_pretrained(f"dpo_{beta}")

# Vérifier si CUDA est disponible et définir le device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Charger le tokenizer et les modèles GPT-2
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
ref_model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

# Charger et réduire le dataset HH d'Anthropic
dataset = load_dataset("Anthropic/hh-rlhf", split="train").select(range(training_size))

# Appliquer le prétraitement pour inverser les labels avec probabilité p
dataset = preprocess_dataset(dataset, probability_p)

# Nettoyer la mémoire avant d'entraîner
torch.cuda.empty_cache()

# Boucle sur les valeurs de beta
for beta in betas:
    print(f"Entraînement pour beta={beta}...")

    # Charger un nouveau modèle GPT-2 pour chaque entraînement
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

    # Initialiser wandb pour chaque valeur de beta
    wandb.init(
        project="DPO-Training",
        config={
            "beta": beta,
            "p": probability_p,
            "model_name": "gpt2",
            "dataset": "Anthropic/hh-rlhf",
            "batch_size": 4,  # Ajustement de la taille pour éviter les problèmes mémoire
            "max_length": 512,
            "max_prompt_length": 256
        },
        name=f"dpo_hh_run_beta_{beta}_{training_size}",
        save_code=True
    )

    # Lancer l'entraînement avec wandb intégré
    train(model, ref_model, dataset, tokenizer, beta)

    # Fin de l'exécution de wandb
    wandb.finish()

    print(f"Entraînement terminé pour beta={beta}. Modèle sauvegardé dans 'dpo_{beta}'.")
