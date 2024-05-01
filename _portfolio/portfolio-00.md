---
title: "The Threat of Transformer Arises — Challenger Mamba (From SSM, HiPPO, S4 to Mamba)"
excerpt: "A reproduce project of Mamba series models<br/><img src='/images/Mamba.png'>"
collection: portfolio
---

### Will upload before May.


#You may find the reproduce work [here](). 

# 1. State Spaces Model

## 1.1 What is State Spaces

Imagine we are walking through a maze. Each small box in the diagram represents a position in the maze and contains some implicit information, such as how far you are from the exit.

![P1](https://dashpulsar.github.io/images/MambaGPT/P1.png)

The above maze can be simplified and modeled as a 'state space representation', where each small box displays:

1. Your current location (current state)
2. Where you can go next (possible future states)
3. Which actions will take you to the next state (e.g., moving right or left)

The variables describing the state (in our example, the X and Y coordinates and the distance to the exit) can be represented as 'state vectors'.

![P2](https://dashpulsar.github.io/images/MambaGPT/P2.png)

## 1.2 What is State Spaces Models (SSM)

SSM (State Space Model) is a model used to describe these state representations and predict their next state based on certain inputs.

Generally, SSMs include the following components:

Mapping input sequences x(t), such as moving left and down in the maze,
To latent state representations h(t), such as the distance to the exit and x/y coordinates,
And deriving predicted output sequences y(t), such as moving left again to reach the exit faster.
However, it does not use discrete sequences (like moving left once), but instead takes continuous sequences as inputs to predict output sequences.