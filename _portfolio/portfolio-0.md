---
title: "The Evolution of Language Generative AI: GPT, GPT-2, GPT-3, and InstructGPT"
excerpt: "A reproduce project of GPTs (GPT-1, GPT-2, GPT-3, InstructGPT and latest chatgpt(3.5))<br/><img src='/images/P0.png'>"
collection: portfolio
---



</br></br></br>

# 1.Introduction
======

In recent years, the field of artificial intelligence has undergone a revolution, at the heart of which are the increasingly powerful Generative Pre-trained Transformer models, known as GPT, developed by leading technology companies like Google and OpenAI. From the first GPT in 2018, to GPT-2, and then to GPT-3 in 2020, followed by InstructGPT, these models have not only pushed the boundaries of natural language processing but have also drastically changed the way we interact with machines.

The original GPT model introduced the Transformer architecture, which optimized the processing of textual data through self-attention mechanisms, allowing the model to capture deep linguistic structures and meanings from vast amounts of text. Its pre-training approach, which involves training on a broad corpus before fine-tuning for specific tasks, significantly enhanced the model's versatility and efficiency.

With the release of GPT-2, OpenAI significantly increased the scale of the model, introducing 1.5 billion parameters, more than ten times the number of its predecessor. This expansion not only enhanced the model's language understanding capabilities but also enabled it to generate longer, more coherent texts persuasive across various styles and themes. The success of GPT-2 demonstrated the tremendous potential of large-scale language models in unsupervised learning.

GPT-3 reached an unprecedented scale with 175 billion parameters. It is capable of performing text generation, language translation, question-answering, summarization, and other complex language tasks, demonstrating near-human capabilities. Additionally, GPT-3's introduction of "few-shot learning," the ability to perform tasks with minimal task-specific training, using just a few examples, greatly expanded the model's range of applications.

The latest model, InstructGPT, is further optimized from GPT-3 and is specifically designed to respond to human instructions. This improvement means that the model is more precise in understanding and executing specific commands, making interactions with AI more intuitive and efficient. The development of InstructGPT marks a significant advancement in human-machine interaction methods, showcasing the future trend of customized AI applications.

Through this blog post, we will explore in detail the developmental journey of these models and their technical details, as well as how these advancements are gradually changing the way we work and live. Starting from the original GPT, we have gone through several phases of development, each marked by technological breakthroughs and innovative ideas, ultimately shaping the intelligent conversational systems we rely on today.

# 2.GPT-1
======

In 2018, the field of NLP was primarily in a phase where deep learning approaches were centered around word2vec and crafting custom deep models for specific tasks. Although pre-trained models like ELMo and BERT had already emerged, their impact was not yet profound. Meanwhile, the rise of deep learning technologies provided new possibilities for handling large datasets. In 2017, researchers from Google published the paper "Attention Is All You Need," introducing the transformer architecture for the first time. This new network structure, through its use of self-attention, could effectively process sequential data, particularly excelling over traditional RNNs and LSTMs in handling long-distance dependencies. Building on the transformer's foundation, and incorporating the pre-training and fine-tuning approaches used in previous models, GPT-1 was developed.[(Radford, Narasimhan, Salimans, and Sutskever, 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

## 2.1 Model Structure
======

During the pre-training phase, GPT selects the decoder part of the transformer as its main module. The transformer is a feature extraction model proposed by Google in 2017. GPT is constructed by stacking multiple layers of transformers to form the entire pre-training model structure.

![P01](https://dashpulsar.github.io/images/P01.png)

Assuming there is a text with each word denoted as \(u_i\), GPT uses the standard language model objective function to maximize the following likelihood function:

$$
L_1(U) = \sum_{i} \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)
$$

Specifically, it predicts the probability of each word \(u_i\), based on the words from \(u_{i-k}\) to \(u_{i-1}\), and the model \(\Theta\). Here, \(k\) represents the context window size; theoretically, the larger \(k\) is, the more context information the model can access, enhancing the model's capabilities.

The model embeds input \(U\) into features to get the input \(h_0\) for the first layer of the transformer, then goes through multiple layers of transformer encoding, and uses the output of the last layer to obtain the current predicted probability distribution:

$$
h_0 = UW_e + W_p \\
h_l = \text{transformer\_block}(h_{l-1}) \\
P(u) = \text{softmax}(h_n W_e^T)
$$

where \(W_e\) is the word embedding matrix, \(W_p\) is the position embedding matrix, \(h_l\) is the output of the \(l\)-th layer of the transformer, \(h_n\) is the output of the last layer of the transformer, and \(n\) is the number of layers.

During the fine-tuning phase, given an input sequence from \(x_1\) to \(x_m\) and a specific downstream task label, the model predicts the probability of \(y\) by inputting the sequence into the pre-trained model, obtaining features \(h_l^m\) of the last token \(x^m\) from the last transformer layer, and then passing it through a prediction layer to get the probability distribution for the corresponding label:

$$
P(y \mid x^1, \dots, x^m) = \text{softmax}(h_l^m W_y)
$$

The objective function for the fine-tuning stage is:

$$
L_2(U) = \sum_{(x,y)} \log P(y \mid x^1, \dots, x^m)
$$

The best results are obtained by joint training of the two objective functions, hence the final objective function is:

$$
L_3(U) = L_2(U) + \lambda \cdot L_1(U)
$$

Compared to RNNs, the Transformer architecture features more structured memory units to address long-distance dependencies and handle longer text sequences, enhancing the robustness of the learned features across various tasks. Originally designed for seq2seq tasks, the Transformer model consists of both an encoder and a decoder; the primary difference is that the encoder can access all information in the sequence—both prior and subsequent context—while the decoder, due to its masking mechanism, can only access its own and preceding text information.

The GPT model employs the Transformer's decoder part, precisely because its pre-training objective is the standard language model objective function, which allows the model to consider only the preceding context when predicting a word, without referencing the subsequent context. In contrast, BERT during its pre-training does not use the standard language model as its objective but opts for a masked language model, which enables it to see all contextual information around a word, similar to a cloze task; therefore, BERT uses the Transformer's encoder component.

Although GPT may not perform as well as BERT on some tasks, its potential for effectively predicting future information may prove greater in the long run. OpenAI's persistent use of the standard language model objective for pre-training, as evidenced by the impressive results of GPT-3 and subsequent ChatGPT, has proven to be both visionary and effective.

## 2.2 Model Training
======

In terms of training data, the original GPT model utilized the BooksCorpus dataset, which contains about 5 GB of text with over 74 million sentences. This dataset comprises approximately 7,000 independent books spanning diverse genres. The primary advantage of selecting this dataset is that book texts often include numerous high-quality, lengthy sentences, which ensures that the model can learn long-distance dependencies.

Some key parameters of the model include:

| Parameters         | Values   |
| --------           | ------   |
| layers             | 12       |
| feature dimension  | 768      |
| head               | 12       |
| total parameters   | 1.17B    |

## 2.3 Downstream Tasks Fine-tuning
======

![P01](https://dashpulsar.github.io/images/P01.png)


As illustrated, the application of the GPT model to four common NLP tasks (text classification, textual entailment, text similarity, and question answering) involves specific constructions for input sequences and designs for the prediction layer.

In general, the sequences are manipulated by adding special "Start" and "Extract" tokens to signify the beginning and end of sequences, respectively, and a "Delim" token is used as necessary to denote separation between segments. In practice, these labels ("Start/Extract/Delim") are represented by specific special symbols. Based on the constructed input sequence for each downstream task, the pretrained GPT model is used for feature encoding, followed by predictions using the feature vector of the last token in the sequence.

It is evident that regardless of how the input sequences or prediction layers vary across different tasks, the core feature extraction module remains constant, demonstrating excellent transferability. This consistency ensures that the deep learning model can adapt to various tasks efficiently without requiring fundamental changes to its architecture.


## 2.4 Summary
======
Here are several point of the GPT-1:

Firstly, it was among the earliest works to propose the use of the pre-train + fine-tuning paradigm in NLP tasks.

Secondly, GPT's experiments demonstrated that the model's accuracy and generalization capabilities continuously improve with the addition of more decoder layers, and there is still room for improvement, as illustrated below:

![P02](https://dashpulsar.github.io/images/P02.png)

Thirdly, the pre-trained model possesses zero-shot capabilities, which can be progressively enhanced with ongoing pre-training, as shown in the following graph:

![P03](https://dashpulsar.github.io/images/P03.png)

To further validate the zero-shot capabilities, OpenAI launched GPT-2 one year after the introduction of GPT-1.


# 3. GPT-2
======



# 4. GPT-3
======


# 5. 


# 6. References

[1] Radford, A., Narasimhan, K., Salimans, T. and Sutskever, I., 2018. Improving language understanding by generative pre-training.

[2] 