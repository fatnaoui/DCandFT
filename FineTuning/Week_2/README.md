
# Group 1: Model Fine-Tuning Project

## Project Overview

This repository focuses on fine-tuning pre-trained language models using two specific models:

1. **unsloth/llama-3-8b-bnb-4bit**
2. **unsloth/mistral-7b-bnb-4bit**

These models are chosen for their efficiency and performance in handling natural language processing tasks with limited computational resources, thanks to their low-bit precision.

## Data Source

The fine-tuning process is conducted using data extracted from the file `book fine-tuning data.csv`. This dataset comprises various text excerpts that have been carefully curated to optimize the models for tasks such as text generation, summarization, and contextual understanding.

## Fine-Tuning Process

- **Model Selection:** We selected the `unsloth/llama-3-8b-bnb-4bit` and `unsloth/mistral-7b-bnb-4bit` models due to their balance between model size and computational efficiency.
- **Data Preparation:** The data extracted from the `book fine-tuning data.csv` was pre-processed to ensure it aligns with the input requirements of the chosen models.
- **Training Configuration:** We utilized specific hyperparameters tailored to the nature of the dataset and the models' architectures to ensure optimal fine-tuning results.
- **Evaluation:** Post fine-tuning, the models were evaluated using a set of metrics to measure their performance in generating coherent, contextually appropriate text.

## Next Steps

- **Model Deployment:** The fine-tuned models will be integrated into our application to enhance its natural language capabilities.
- **Further Optimization:** Based on the initial results, additional rounds of fine-tuning may be conducted to further refine the models' outputs.

This fine-tuning effort is expected to significantly improve the performance of the models on specific tasks related to our project goals.

---

Feel free to contribute, report issues, or suggest improvements!
