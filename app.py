import streamlit as st
import torch
import torch.nn as nn
import sentencepiece as spm
import unicodedata
import math
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Urdu Conversational Chatbot",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for RTL support
st.markdown("""
<style>
    .stTextInput input {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', 'Arial', sans-serif;
    }
    .stChatMessage {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2em;
    }
</style>
""", unsafe_allow_html=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture (CORRECTED VERSION)
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
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_linear(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):  # FIXED: Changed d__ff to d_ff
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # FIXED: Changed d__ff to d_ff
        self.linear2 = nn.Linear(d_ff, d_model)  # FIXED: Changed d__ff to d_ff
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.ff(x)
        return self.norm2(x + self.dropout(ff))

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
        return self.norm3(x + self.dropout(ff))

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=2, num_enc_layers=2, num_dec_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout) for _ in range(num_enc_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dropout) for _ in range(num_dec_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.dropout(self.pos_enc(self.embedding(src) * math.sqrt(self.embedding.embedding_dim)))
        tgt = self.dropout(self.pos_enc(self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)))
        enc_out = src
        for layer in self.enc_layers:
            enc_out = layer(enc_out, src_mask)
        dec_out = tgt
        for layer in self.dec_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
        return self.linear(dec_out)

# Utility functions
def normalize_urdu(text):
    if not isinstance(text, str):
        return ""
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = text.replace('ﺁ', 'آ').replace('ﺂ', 'آ').replace('ﺃ', 'أ').replace('ﺄ', 'أ')
    text = text.replace('ﻳ', 'ی').replace('ﻴ', 'ی').replace('ﻰ', 'ی').replace('ﻱ', 'ی').replace('ﻲ', 'ی')
    return text.strip()

def create_src_mask(src):
    return (src != pad_token).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt):
    seq_len = tgt.size(1)
    no_future_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(device) == 0
    pad_mask = (tgt != pad_token).unsqueeze(1).unsqueeze(2)
    return no_future_mask & pad_mask

# Global variables
pad_token = 0
bos_token = 1
eos_token = 2
mask_token = 3

# Initialize components
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Load tokenizer
        tokenizer = spm.SentencePieceProcessor()
        
        # Try to load from different possible locations
        tokenizer_paths = [
            'urdu_spm.model',
            './urdu_spm.model',
            'models/urdu_spm.model',
            './models/urdu_spm.model'
        ]
        
        tokenizer_loaded = False
        for path in tokenizer_paths:
            if os.path.exists(path):
                tokenizer.load(path)
                tokenizer_loaded = True
                st.success(f"Loaded tokenizer from: {path}")
                break
        
        if not tokenizer_loaded:
            st.error("Tokenizer file not found. Please ensure 'urdu_spm.model' is in your repository.")
            return None, None
        
        # Initialize model
        vocab_size = tokenizer.get_piece_size()
        model = Transformer(vocab_size, d_model=256, num_heads=2, num_enc_layers=2, num_dec_layers=2, dropout=0.1).to(device)
        
        # Update token IDs based on actual tokenizer
        global bos_token, eos_token, mask_token
        bos_token = tokenizer.piece_to_id('<s>') or 1
        eos_token = tokenizer.piece_to_id('</s>') or 2
        mask_token = tokenizer.piece_to_id('<mask>') or 3
        
        # Try to load model from different possible locations
        model_paths = [
            'best_model.pt',
            './best_model.pt',
            'models/best_model.pt', 
            './models/best_model.pt',
            'pretrain_model.pt',
            './pretrain_model.pt'
        ]
        
        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=device))
                model_loaded = True
                st.success(f"Loaded model from: {path}")
                break
        
        if not model_loaded:
            st.warning("No pre-trained model found. Using randomly initialized weights.")
        
        model.eval()
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def tokenize(text):
    return tokenizer.encode_as_ids(text)

def generate_response(input_text, max_len=50):
    if model is None or tokenizer is None:
        return "Model not loaded properly. Please check the error messages above."
    
    try:
        input_text = normalize_urdu(input_text.strip())
        if not input_text:
            return "Please enter some text."
            
        tokens = tokenize(input_text)
        if not tokens:
            return "Could not tokenize input text."
            
        src = torch.tensor([[bos_token] + tokens + [eos_token]]).to(device)
        src_mask = create_src_mask(src)
        tgt = torch.tensor([[bos_token]]).to(device)
        
        for _ in range(max_len):
            tgt_mask = create_tgt_mask(tgt)
            out = model(src, tgt, src_mask, tgt_mask)
            pred = out[:, -1, :].argmax(-1).unsqueeze(0)
            tgt = torch.cat([tgt, pred], dim=1)
            if pred.item() == eos_token:
                break
        
        response = tokenizer.decode(tgt[0][1:].cpu().tolist())
        return response if response else "No response generated."
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main app
def main():
    st.markdown('<div class="title">💬 Urdu Conversational Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Experience AI-powered Urdu conversations</div>', unsafe_allow_html=True)
    
    # Load model (this will show status messages)
    global model, tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("اپنا پیغام یہاں لکھیں (Type your message here)"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("جواب تیار ہو رہا ہے... (Generating response...)"):
                response = generate_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This Urdu Chatbot uses a Transformer model trained on Urdu conversational data.
        
        **Features:**
        - Natural Urdu conversations
        - Context-aware responses
        - Real-time interaction
        
        **Instructions:**
        1. Type your message in Urdu
        2. Press Enter to send
        3. The bot will respond in Urdu
        """)
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        st.header("Model Info")
        if tokenizer:
            st.write(f"Vocabulary Size: {tokenizer.get_piece_size()}")
        st.write(f"Device: {device}")
        st.write("Model: Transformer (2 encoder/decoder layers)")
        
        # Debug info
        with st.expander("Debug Information"):
            st.write("Model loaded:", model is not None)
            st.write("Tokenizer loaded:", tokenizer is not None)
            if tokenizer:
                st.write("Sample tokens:", tokenizer.encode_as_ids("سلام"))

# Global model variables
model = None
tokenizer = None

if __name__ == "__main__":
    main()
