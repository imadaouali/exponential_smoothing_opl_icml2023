# [Exponential Smoothing for Off-Policy Learning](https://proceedings.mlr.press/v202/aouali23a/aouali23a.pdf)

Experiments for the paper [Exponential Smoothing for Off-Policy Learning](https://proceedings.mlr.press/v202/aouali23a/aouali23a.pdf)

[Imad AOUALI](https://www.iaouali.com/) (Criteo and ENSAE-CREST), Victor-Emmanuel Brunel (ENSAE-CREST), David Rohde (Criteo), Anna Korba (ENSAE-CREST)

## Abstract

Off-policy learning (OPL) aims at finding improved policies from logged bandit data, often by minimizing the inverse propensity scoring (IPS) estimator of the risk. In this work, we investigate a smooth regularization for IPS, for which we derive a two-sided PAC-Bayes generalization bound. The bound is tractable, scalable, interpretable and provides learning certificates. In particular, it is also valid for standard IPS without making the assumption that the importance weights are bounded. We demonstrate the relevance of our approach and its favorable performance through a set of learning tasks. Since our bound holds for standard IPS, we are able to provide insight into when regularizing IPS is useful. Namely, we identify cases where regularization might not be needed. This goes against the belief that, in practice, clipped IPS often enjoys favorable performance than standard IPS in OPL.

## Repository Structure

This repository is structured as follows

- `utils.ipynb`
meTS experiments on synthetic linear bandit problems

- `policies.ipynb`
meTS experiments on MovieLens dataset with linear rewards

- `meTS-Log.ipynb` 
meTS experiments on synthetic logistic bandit problems

- `meTS-Log-MovieLens.ipynb`
meTS experiments on MovieLens dataset with logistic rewards

- `ratings.dat`
[MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/)

[imad-email]: mailto:imadaouali9@gmail.com 

## Acknowledgement
The code for the baselines was provided by Otmane Sakhi, Pierre Alquier, Nicolas Chopin, [PAC-Bayesian Offline Contextual Bandits With Guarantees](https://proceedings.mlr.press/v202/sakhi23a/sakhi23a.pdf). Many thanks to them.
