# GPT-2 Transformer Model from Scratch

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

A sleek implementation of a GPT-2-based Large Language Model (LLM) built from scratch using Python and PyTorch, featuring transformer blocks, multi-head self-attention, and optimized text generation.

## 🌟 Features
- **GPT-2 Architecture**: Transformer blocks with multi-head attention, GELU-activated feed-forward networks, and layer normalization.
- **Efficient Tokenization**: Byte-pair encoding via TikToken, handling 5,104 tokens.
- **Custom Data Pipeline**: Sliding window dataset and dataloader for seamless training.
- **Text Generation**: Deterministic and probabilistic outputs with temperature scaling and top-k sampling.
- **Pretraining**: 10 epochs with loss perplexity for robust performance.

## 🚀 Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/Rohitw3code/LLM-from-scratch.git
   cd LLM-from-scratch
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   **requirements.txt**:
   ```
   torch>=2.0.0
   tiktoken>=0.7.0
   ```
3. Add your text dataset to the project directory.

## 🛠️ Usage
- **Prepare Data**: Tokenize text using TikToken's GPT-2 encoder (see `previous_chapters.py`).
- **Train**: Configure `GPT_CONFIG_124M` and run the training loop in `4_Pretraining_on_unlabeled_Data.ipynb`.
- **Generate Text**: Use `generate_text_simple` for text generation with customizable sampling.
- **Evaluate**: Monitor loss perplexity during training.

## 📂 Project Structure
```
LLM-from-scratch/
├── previous_chapters.py          # Model, dataset, and dataloader
├── 4_Pretraining_on_unlabeled_Data.ipynb  # Training and evaluation
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

## 🧠 Model Architecture
- **Embeddings**: Token and positional embeddings for input processing.
- **Multi-Head Attention**: Captures complex dependencies with 12 heads.
- **Feed-Forward Networks**: GELU activation for non-linearity.
- **Transformer Blocks**: 12-layer stack with 124M parameters.
- **Config**: 50,257 vocab size, 1,024 context length.

## 🎨 Text Generation
- **Deterministic**: Uses `torch.argmax` for highest-probability tokens.
- **Probabilistic**: Temperature scaling and top-k sampling for diverse outputs.

## 📈 Training
- **Dataset**: 5,104 tokens via TikToken.
- **Training**: 10 epochs, batch size 4, sequence length 256, stride 128.
- **Evaluation**: Loss perplexity.

## 🔮 Future Enhancements
- Add top-p sampling.
- Support fine-tuning for specific tasks.
- Scale to larger model configurations.

## 🤝 Contributing
1. Fork the repo.
2. Create a feature branch (`git checkout -b feature`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push (`git push origin feature`).
5. Open a pull request.

## 📜 License
MIT License. See [LICENSE](LICENSE).

## 🙌 Acknowledgments
Inspired by OpenAI's GPT-2 and *LLMs from Scratch* by Sebastian Raschka.
