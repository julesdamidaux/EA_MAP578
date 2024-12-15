# EA_MAP578
This repository contains the code used to plot the graph we showed in our presentation/report of the EA.

The two files dpo.py and dpo_poison.py are used to train the models using dpo with or without data poisoning.

The file distance.py is used to compute the distances of the dpo models to the reference model GPT2.

The file compute_winrate_v1.py computes the winrates of the dpo models vs GPT2, using the first method which just ask to the judge pipeline which answer is the best, which gave us bad results (misalignement of the reward compared to the objective).

The file compute_winrate_v2.py computes the winrates of the dpo models vs GPT2, using another method (comparing the helpfulness and harmfulness of the answers). This method seemed to give us consistent result with what we expected.
