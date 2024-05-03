---
title: "The Threat of Transformer Arises — Challenger Mamba (From SSM, HiPPO, S4 to Mamba)"
excerpt: "A reproduce project of Mamba series models<br/><img src='/images/Mamba.png'>"
collection: portfolio
---

### Will upload before May.


#You may find the reproduce work [here](). 

# 1 State Spaces Model

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

1. Mapping input sequences x(t), such as moving left and down in the maze,
2. To latent state representations h(t), such as the distance to the exit and x/y coordinates,
3. And deriving predicted output sequences y(t), such as moving left again to reach the exit faster.

However, it does not use discrete sequences (like moving left once), but instead takes continuous sequences as inputs to predict output sequences.

![P3](https://dashpulsar.github.io/images/MambaGPT/P3.png)

SSMs assume that dynamic systems, such as an object moving in 3D space, can be predicted from its state at time t through two equations.

![P4](https://dashpulsar.github.io/images/MambaGPT/P4.png)

By solving these equations, it is possible to predict the state of the system based on observed data (input sequences and previous states). Therefore, the above equations represent the core idea of the State Space Model (SSM).

## 1.3 State Equation and Output Equation

The state equation describes how the state changes (through matrix A) based on how the input influences the state (through matrix B).

![P5](https://dashpulsar.github.io/images/MambaGPT/P5.png)

As we saw before, h(t) refers to our latent state representation at any given time t, and x(t) refers to some input.

The output equation describes how the state is translated to the output (through matrix C) and how the input influences the output (through matrix D).

![P6](https://dashpulsar.github.io/images/MambaGPT/P6.png)

## 1.4 Fata Flow Detail

Visualizing above two equations gives us the following architecture:

![P7](https://dashpulsar.github.io/images/MambaGPT/P7.png)


Assume we have some input signal x(t), this signal first gets multiplied by matrix B which describes how the inputs influence the system.

![P8](https://dashpulsar.github.io/images/MambaGPT/P8.png)

The updated state (akin to the hidden state of a neural network) is a latent space that contains the core “knowledge” of the environment. We multiply the state with matrix A which describes how all the internal states are connected as they represent the underlying dynamics of the system.

![P9](https://dashpulsar.github.io/images/MambaGPT/P9.png)

Matrix A is applied before creating the state representations and is updated after the state representation has been updated.

Then, we use matrix C to describe how the state can be translated to an output.

![P10](https://dashpulsar.github.io/images/MambaGPT/P10.png)

Finally, we can make use of matrix D to provide a direct signal from the input to the output. This is also often referred to as a skip-connection.

![P11](https://dashpulsar.github.io/images/MambaGPT/P11.png)

Since matrix D is similar to a skip-connection, the SSM is often regarded as the following without the skip-connection.

![P12](https://dashpulsar.github.io/images/MambaGPT/P12.png)

Going back to our simplified perspective, we can now focus on matrices A, B, and C as the core of the SSM.

![P13](https://dashpulsar.github.io/images/MambaGPT/P13.png)

We can update the original equations (and add some pretty colors) to signify the purpose of each matrix as we did before.

![P14](https://dashpulsar.github.io/images/MambaGPT/P14.png)

Together, these two equations aim to predict the state of a system from observed data. Since the input is expected to be continuous, the main representation of the SSM is a continuous-time representation.

# 2 From SSM to S4

Three-step upgrade from SSM to S4: discretized SSM, cyclic/convolutional representation, long sequence processing based on HiPPO

## 2.1 From a Continuous to a Discrete Signal

In addition to continuous inputs, discrete inputs (such as text sequences) are also commonly encountered. However, even when trained on discrete data, an SSM can still learn the underlying continuous information. This is because, for an SSM, a sequence is merely a sampling of a continuous signal, or in other words, the continuous signal model is a generalization of the discrete sequence model.

![P15](https://dashpulsar.github.io/images/MambaGPT/P15.png)


The Zero-order hold technique can be used to discretize the model, thus handling discrete signals.

1. Initially, when a discrete signal is received, its value is maintained until a new discrete signal is received. This operation results in the creation of a continuous signal that the SSM can utilize.

![P16](https://dashpulsar.github.io/images/MambaGPT/P16.png)

2. The duration for which this value is held is represented by a new learnable parameter known as the step size (siz) — Δ, which signifies the phased hold (resolution) of the input.

3. With the continuous input signal available, continuous outputs can be generated, and values are sampled based solely on the input’s time step size.

![P17](https://dashpulsar.github.io/images/MambaGPT/P17.png)


These sampled values become our discrete outputs, and for matrices A and B, the zero-order hold can be applied in the following way: 

Discretized matrix A

$$
\(\bar{A} = \exp(\Delta A)\)
$$

Discretized matrix B

$$
\(\bar{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B\)
$$


They collectively enable us to transition from a continuous SSM to a discretized SSM represented by the formulas. The model no longer represents a function to function \\(x(t) \rightarrow y(t)\\), but rather a sequence to sequence \\(x_k \rightarrow y_k\\):

![P18](https://dashpulsar.github.io/images/MambaGPT/P18.png)


Here, matrices A and B now represent the discrete parameters of the model.

We use k instead of t to denote discrete time steps, making it clearer when we refer to continuous SSM versus discrete SSM.

Note: During training still maintain the continuous form of matrix A, not the discretized version. The continuous representation is discretized during the training process.

## 2.2 The Recurrent Representation

The discrete SSM allows the problem to be reformulated using discrete time steps. At each time step, we compute how the current input \\(\(Bx_k\)\\) affects the previous state \\(\(Ah_{k-1}\)\\), and then calculate the predicted output \\(\(Ch_k\)\\).

![P19](https://dashpulsar.github.io/images/MambaGPT/P19.png)


If we expand y2 we can get the following:

$$
y_2 = Ch_2
    = C(Ah_1 + Bx_2)
    = C(A(Ah_0 + Bx_1) + Bx_2)
    = C(A(A \cdot Bx_0 + Bx_1) + Bx_2)
    = C(A \cdot A \cdot Bx_0 + A \cdot Bx_1 + Bx_2)
    = C \cdot A^2 \cdot Bx_0 + C \cdot A \cdot B \cdot x_1 + C \cdot Bx_2
$$


## 2.3 The Convolution Representation

Another representation that we can use for SSMs is that of convolutions. Remember from classic image recognition tasks where we applied filters (kernels) to derive aggregate features:

![P20](https://dashpulsar.github.io/images/MambaGPT/P20.png)

Since we are dealing with text and not images, we need a 1-dimensional perspective instead:

![P21](https://dashpulsar.github.io/images/MambaGPT/P21.png)

The kernel that we use to represent this “filter” is derived from the SSM formulation:

![P22](https://dashpulsar.github.io/images/MambaGPT/P22.png)

Let’s explore how this kernel works in practice. Like convolution, we can use our SSM kernel to go over each set of tokens and calculate the output:

![P23](https://dashpulsar.github.io/images/MambaGPT/P23.png)

This also illustrates the effect padding might have on the output. I changed the order of padding to improve the visualization but we often apply it at the end of a sentence.

In the next step, the kernel is moved once over to perform the next step in the calculation:

![P24](https://dashpulsar.github.io/images/MambaGPT/P24.png)

In the final step, we can see the full effect of the kernel:

![P25](https://dashpulsar.github.io/images/MambaGPT/P25.png)

A major benefit of representing the SSM as a convolution is that it can be trained in parallel like Convolutional Neural Networks (CNNs). However, due to the fixed kernel size, their inference is not as fast and unbounded as RNNs.

If we separate the parameters and inputs of the formula, we get:

$$
y_3 = \begin{pmatrix}
    CAAA & CAAB & CAB & CB
\end{pmatrix}
\begin{pmatrix}
    x_0 \\
    x_1 \\
    x_2 \\
    x_3
\end{pmatrix}
$$

Since the three discrete parameters A, B, and C are constants, we can pre-compute the left-hand side vector and save it as a convolution kernel. This provides us with a straightforward method to compute \\(y\\) at very high speeds using convolution, as shown in the following two equations:

$$

\[K = (CB \quad CAB \quad \dots \quad CA^kB)\]

\[y = K \ast x\]

$$


## 2.4 Continuous, Recurrent, and Convolutional Representations

These three representations, continuous, recurrent, and convolutional all have different sets of advantages and disadvantages:

![P26](https://dashpulsar.github.io/images/MambaGPT/P26.png)

One of the main benefits of representing the SSM as a convolution is that it can be trained in parallel similar to convolutional neural networks (CNNs). However, due to the fixed size of the kernel, their inference is not as fast as that of RNNs.




Interestingly, we can now use a recurrent SSM for efficient inference and a convolutional SSM for parallel training.

With these representations, we can use a clever trick: choose the representation based on the task. During training, we use the convolutional representation that can be parallelized, and during inference, we use the efficient recurrent representation:

![P27](https://dashpulsar.github.io/images/MambaGPT/P27.png)

This model is called the Linear State Space Layer (LSSL).

These representations share an important property, linear time-invariance (LTI). LTI dictates that the SSM parameters A, B, and C are fixed for all time steps. This means that for every token generated by the SSM, the matrices A, B, and C are the same.

In other words, no matter what sequence you provide to the SSM, the values of A, B, and C remain unchanged. We have a static representation that does not recognize content.

Before we explore how Mamba addresses this issue, let's discuss the final piece of this puzzle: matrix A.

## 2.5 HiPPO, The solution to the long-distance dependency problem

Matrix A can be considered one of the most crucial aspects of the SSM formula. As we have seen in the recurrent representation, it captures information about previous states to construct new states.
 \\(h_k = \overline{A} h_{k-1} + \overline{B} x_k\\), and when k = 5, then \\(h_5 = \overline{A} h_{4} + \overline{B} x_5\\).

![P28](https://dashpulsar.github.io/images/MambaGPT/P28.png)

In essence, matrix A produces the hidden state:

![P29](https://dashpulsar.github.io/images/MambaGPT/P29.png)


Since matrix A only remembers the previous few tokens and captures the differences between every token seen so far, especially in the context of the recurrent representation, as it only looks back at previous states. Therefore, we need a way to create matrix A that can retain a longer memory, namely High-order Polynomial Projection Operators (HiPPO).

HiPPO attempts to compress all the input signals seen so far into a vector of coefficients.

![P30](https://dashpulsar.github.io/images/MambaGPT/P30.png)


It uses matrix A to construct a state representation that captures recent tokens well and attenuates older tokens, producing an optimal solution for state matrix A through function approximation. The formula can be represented as follows:

![P31](https://dashpulsar.github.io/images/MambaGPT/P31.png)

As following table:

![P32](https://dashpulsar.github.io/images/MambaGPT/P32.png)

Building matrix A using HiPPO was shown to be much better than initializing it as a random matrix. As a result, it more accurately reconstructs newer signals (recent tokens) compared to older signals (initial tokens).

The idea behind the HiPPO Matrix is that it produces a hidden state that memorizes its history.

Mathematically, it does so by tracking the coefficients of a Legendre polynomial which allows it to approximate all of the previous history.


HiPPO was then applied to the recurrent and convolution representations that we saw before to handle long-range dependencies. The result was Structured State Space for Sequences (S4), a class of SSMs that can efficiently handle long sequences.

It consists of three parts:

1. State Space Models

2. HiPPO for handling long-range dependencies

3. Discretization for creating recurrent and convolution representations

![P33](https://dashpulsar.github.io/images/MambaGPT/P33.png)

This class of SSMs has several benefits depending on the representation you choose (recurrent vs. convolution). It can also handle long sequences of text and store memory efficiently by building upon the HiPPO matrix.


## 2.6 How S4 works

### 2.6.1 Mapping Input to State, Optimize long sequence processing

Sequence data is generally discrete, such as text, images, and DNA. However, there are many continuous types of data in real life, such as audio and video. A significant characteristic of audio and video signals is their extremely long context windows.

Transformers often fail on long contexts, and attention mechanisms are not particularly adept at tasks with such extensive context lengths. This has led to various improvements to the attention mechanism, such as flash attention, etc. Even so, they typically handle contexts up to about 32K in length, and are powerless when faced with sequence lengths of 1 million. S4, on the other hand, excels at these types of tasks.



