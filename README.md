# Project 10: RNN Text Generation with LSTM Layers

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-LSTM-red)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)]()

A Recurrent Neural Network (RNN) built with stacked LSTM layers for character-level text generation, trained on Shakespeare's works. The model learns patterns in sequential text data and generates new, stylistically consistent text from a seed prompt.

---

## 🎯 Project Goal

To design, train, and evaluate an LSTM-based RNN for sequence modelling and text generation — demonstrating how LSTMs capture long-range dependencies in sequential data that simple RNNs struggle with.

---

## ✨ What This Project Does

1. Downloads Shakespeare's text corpus (~1M characters) automatically
2. Preprocesses it into character-level sequences for training
3. Trains a stacked LSTM model to predict the next character given 100 previous characters
4. Generates new Shakespearean-style text from seed prompts like `"ROMEO: "` or `"HAMLET: To be "`
5. Demonstrates the effect of **temperature** on generation creativity vs coherence

---

## 🧠 Model Architecture

```
Input: (batch_size, 100, vocab_size)  ← one-hot encoded sequences
│
├── LSTM(128, return_sequences=True)
├── Dropout(0.2)
├── LSTM(128)
├── Dropout(0.2)
└── Dense(vocab_size, activation='softmax')  ← probability over all characters
```

| Layer | Output Shape | Parameters |
|---|---|---|
| LSTM 1 (128 units) | (batch, 100, 128) | ~167K |
| Dropout (0.2) | — | — |
| LSTM 2 (128 units) | (batch, 128) | ~131K |
| Dropout (0.2) | — | — |
| Dense (softmax) | (batch, vocab_size) | ~8K |

**Compiled with:**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metric: Accuracy

---

## 📊 Training Configuration

| Parameter | Value |
|---|---|
| Sequence Length | 100 characters |
| Step Size | 3 (sliding window) |
| Max Training Sequences | 40,000 |
| Train / Val Split | 90% / 10% |
| Batch Size | 128 |
| Epochs | 50 |
| Early Stopping | Monitored on val_loss |

---

## 🌡️ Temperature in Text Generation

Temperature controls the randomness of generated text by scaling the model's output probabilities before sampling.

| Temperature | Effect |
|---|---|
| 0.5 | Conservative, predictable, repetitive |
| 0.8 | More fluent, slightly creative |
| 1.0 | Balanced — default setting |
| 1.2 | Creative, surprising, less coherent |

---

## 📁 Project Structure

```
ML_Task_10/
│
├── ML_Task_10_Tayyabah_Rehman.ipynb   # Main notebook (full pipeline)
├── shakespeare.txt                     # Downloaded automatically at runtime
├── lstm_training_history.png           # Loss and accuracy plots
├── generated_text.txt                  # Saved generated text samples
├── best_lstm_model.keras               # Saved model weights
└── README.md                           # This file
```

---

## 🔧 Setup & Installation

### 1. Clone or download this repository

```bash
git clone https://github.com/your-username/ML_Task_10_RNN_TextGeneration.git
cd ML_Task_10_RNN_TextGeneration
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install tensorflow numpy matplotlib
```

### 4. Run the notebook

Open `ML_Task_10_Tayyabah_Rehman.ipynb` in Jupyter or VS Code and run all cells top to bottom. The Shakespeare dataset downloads automatically.

---

## 🚀 Usage

The notebook runs end-to-end with no manual data downloads. Key sections:

**Generate text after training:**
```python
result = generate_text(model, seed_text="ROMEO: ", num_chars=250, temperature=1.0)
print(result)
```

**Try different temperatures:**
```python
for temp in [0.5, 0.8, 1.0, 1.2]:
    print(f"\n--- Temperature: {temp} ---")
    print(generate_text(model, "HAMLET: To be ", num_chars=150, temperature=temp))
```

---

## 📋 Requirements

```
tensorflow >= 2.x
numpy
matplotlib
```

Standard library: `urllib`, `gc`, `warnings`

No additional packages needed — the Shakespeare text downloads automatically via `urllib`.

---

## 📝 Key Concepts Demonstrated

**Why LSTMs over vanilla RNNs?**
Standard RNNs suffer from vanishing gradients — they lose context over long sequences. LSTM cells solve this with three gating mechanisms (input, forget, output gates) that selectively remember or forget information across many timesteps. With a sequence length of 100 characters, this matters significantly.

**Why character-level (not word-level)?**
Character-level models have a small fixed vocabulary (~65 unique characters vs thousands of words), making training more tractable on a small corpus. They can also generate novel word forms the training data never contained.

**Why Dropout?**
Two Dropout(0.2) layers after each LSTM reduce overfitting by randomly deactivating 20% of neurons during training, improving generalisation to validation data.

---

## 👤 Author

**Tayyabah Rehman** 
MPhil Artificial Intelligence, University of the Punjab, Lahore  
GitHub: [@Tayyabah-Rehman](https://github.com/Tayyabah-Rehman)
