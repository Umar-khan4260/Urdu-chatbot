import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import unicodedata
import re
import math
import pickle
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ===== MODEL ARCHITECTURE =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn))
        cross_attn = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attn))
        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=2, num_layers=2, dropout=0.1, max_len=22):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_enc(src.transpose(0, 1)).transpose(0, 1)
        src = self.dropout(src)

        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.pos_enc(tgt.transpose(0, 1)).transpose(0, 1)
        tgt = self.dropout(tgt)

        enc_out = src
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        dec_out = tgt
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        return self.linear(dec_out)

# ===== UTILITY FUNCTIONS =====
def normalize_urdu(text):
    if not isinstance(text, str):
        return ""
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[أإ]', 'ا', text)
    text = re.sub(r'ے', 'ی', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def tokenize(text):
    return text.split()

def create_src_mask(src, pad_idx=0):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt, pad_idx=0):
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool().to(tgt.device)
    return tgt_pad_mask & tgt_sub_mask

# ===== MODEL LOADING =====
@st.cache_resource
def load_model_and_vocab():
    try:
        # First, check if files exist
        if not os.path.exists('best_model.pth'):
            st.error("Model file 'best_model.pth' not found!")
            return None, None, None
        
        # Create a sample vocabulary (you should replace this with your actual vocabulary)
        # This is a temporary fix - you need to save your actual vocabulary during training
        sample_vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + [f'word_{i}' for i in range(252)]
        word_to_idx = {word: idx for idx, word in enumerate(sample_vocab)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        # Initialize model
        vocab_size = len(sample_vocab)
        model = Transformer(
            vocab_size=vocab_size,
            d_model=256,
            num_heads=2,
            num_layers=2,
            dropout=0.1
        )
        
        # Load model weights
        device = 'cpu'
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model, word_to_idx, idx_to_word
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def generate_response(model, src, word_to_idx, idx_to_word, max_len=20, device='cpu'):
    if model is None:
        return "Model not loaded properly."
    
    model.eval()
    src = src.to(device)
    src_mask = create_src_mask(src, word_to_idx['<PAD>']).to(device)

    tgt = torch.tensor([[word_to_idx['<SOS>']]], dtype=torch.long).to(device)

    generated_tokens = []
    for _ in range(max_len):
        tgt_mask = create_tgt_mask(tgt, word_to_idx['<PAD>']).to(device)
        with torch.no_grad():
            output = model(src, tgt, src_mask, tgt_mask)
        next_token = output[:, -1, :].argmax(dim=-1).item()
        if next_token == word_to_idx['<EOS>']:
            break
        generated_tokens.append(next_token)
        tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)

    response = [idx_to_word.get(idx, '<UNK>') for idx in generated_tokens if idx not in [word_to_idx['<SOS>'], word_to_idx['<EOS>']]]
    return ' '.join(response)

def text_to_indices(text, word_to_idx, max_len=20):
    if word_to_idx is None:
        return torch.tensor([[0]], dtype=torch.long)
    
    tokens = tokenize(normalize_urdu(text))
    indices = [word_to_idx['<SOS>']] + [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens] + [word_to_idx['<EOS>']]
    indices = indices[:max_len + 2]
    indices = indices + [word_to_idx['<PAD>']] * (max_len + 2 - len(indices))
    return torch.tensor([indices], dtype=torch.long)

# ===== STREAMLIT APP =====
def main():
    st.set_page_config(
        page_title="Urdu Conversational Chatbot",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("🕌 Urdu Conversational Chatbot")
    st.markdown("""
    Welcome to the Urdu Conversational Chatbot! This AI-powered chatbot can have conversations in Urdu.
    Type your message in Urdu and get a response!
    """)
    
    # Load model
    with st.spinner('Loading model... This may take a moment.'):
        model, word_to_idx, idx_to_word = load_model_and_vocab()
    
    if model is None:
        st.error("""
        **Failed to load the model.** This could be because:
        - The model file is missing or corrupted
        - There's a version mismatch with PyTorch
        - The model architecture doesn't match
        
        Please check that 'best_model.pth' is in your repository.
        """)
        return
    
    # Chat interface
    st.subheader("💬 Chat with the Bot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "سلام! میں آپ کی کیسے مدد کر سکتی ہوں؟"}
        ]
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("Type your message in Urdu..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner('Thinking...'):
            try:
                input_tensor = text_to_indices(prompt, word_to_idx)
                response = generate_response(model, input_tensor, word_to_idx, idx_to_word)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.chat_message("assistant").markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot uses a Transformer model trained on Urdu conversational data.
        
        **Features:**
        - Natural Urdu conversations
        - Context-aware responses
        - Real-time interaction
        
        **How to use:**
        1. Type your message in Urdu
        2. Press Enter or click Send
        3. Wait for the AI response
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "سلام! میں آپ کی کیسے مدد کر سکتی ہوں؟"}
            ]
            st.rerun()
        
        st.markdown("---")
        st.markmary("**Technical Details:**")
        st.code("""
Model: Transformer
Layers: 2
Heads: 2
Embedding: 256
Vocabulary: 256 tokens
        """)

if __name__ == "__main__":
    main()
