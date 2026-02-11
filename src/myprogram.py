#!/usr/bin/env python
import os
import string
import random
import bz2
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention Is All You Need'
    Lecture: Transformers - breaks permutation invariance of self-attention
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Transformer decoder block with causal masking
    Lecture: Transformers - self-attention + FFN with residuals and layer norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Feedforward network (expands to 4x, then projects back)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU from GPT-2+
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (from Neural Networks lecture)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-layer norm (modern practice)
        # Self-attention with residual connection
        normed = self.ln1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        
        # FFN with residual connection
        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x


class CharTransformer(nn.Module):
    """
    Transformer-based character-level language model
    Combines concepts from Transformers and Neural Networks lectures
    """
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4, d_ff=1024, dropout=0.1, max_len=512):
        super(CharTransformer, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)  # Final layer norm
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize weights (He initialization for better gradient flow)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using techniques from Neural Networks lecture"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Create causal mask (from Transformers lecture - prevent attending to future)
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)  # Scale embeddings
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.fc_out(x)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class CharLSTM(nn.Module):
    """
    Improved LSTM with better regularization
    Kept as fallback option
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)  # Added layer norm
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.ln(lstm_out)  # Layer normalization
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden


class CharDataset(Dataset):
    """Dataset for character sequences"""
    def __init__(self, text_sequences, char2idx, sequence_length=100):
        self.char2idx = char2idx
        self.sequence_length = sequence_length
        
        # Combine all text sequences into one string
        print(f"Combining {len(text_sequences)} text sequences...")
        combined_text = '\n'.join(text_sequences)
        print(f"Total characters in combined text: {len(combined_text)}")
        
        # Convert to indices
        self.data = [char2idx.get(c, char2idx['<UNK>']) for c in combined_text]
    
    def __len__(self):
        return max(0, len(self.data) - self.sequence_length)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + 1:idx + self.sequence_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class MyModel:
    """
    LSTM-based character predictor
    """

    # Paths to check for the training dump (e.g. Wikipedia dump.xml.bz2)
    TRAIN_DUMP_PATHS = [
        'data/dump.xml.bz2',
        'dump.xml.bz2',
        '/job/data/dump.xml.bz2',
    ]

    def __init__(self, vocab_size=None, char2idx=None, idx2char=None, use_transformer=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.use_transformer = use_transformer
        
        if vocab_size is not None:
            if use_transformer:
                # Transformer-based model (better for long-range dependencies)
                self.model = CharTransformer(
                    vocab_size=vocab_size,
                    d_model=256,
                    num_heads=8,
                    num_layers=4,
                    d_ff=1024,
                    dropout=0.1
                ).to(self.device)
            else:
                # LSTM fallback
                self.model = CharLSTM(vocab_size).to(self.device)
        else:
            self.model = None

    @classmethod
    def load_training_data(cls, max_texts=None):
        """
        Load training data from a bz2-compressed XML dump (e.g. Wikipedia).
        Looks for dump.xml.bz2 in data/, current dir, or /job/data (Docker).
        Returns a list of text strings (one per article/page).
        If max_texts is set, only the first that many texts are returned (for debugging).
        """
        dump_path = None
        for path in cls.TRAIN_DUMP_PATHS:
            if os.path.isfile(path):
                dump_path = path
                break
        if dump_path is None:
            print("Warning: No training dump found. Returning empty dataset.")
            return []

        print(f"Loading training data from {dump_path}...")
        texts = []
        with bz2.open(dump_path, 'rt', encoding='utf-8', errors='replace') as f:
            in_text = False
            current = []
            for line in f:
                if not in_text:
                    if '<text' in line:
                        in_text = True
                        start = line.find('>') + 1
                        if '</text>' in line[start:]:
                            end = line.find('</text>', start)
                            texts.append(line[start:end])
                            if max_texts is not None and len(texts) >= max_texts:
                                print(f"Loaded {len(texts)} texts (max_texts limit reached)")
                                return texts
                            in_text = False
                        else:
                            current = [line[start:].rstrip('\n')]
                else:
                    if '</text>' in line:
                        end = line.find('</text>')
                        current.append(line[:end])
                        texts.append('\n'.join(current))
                        if max_texts is not None and len(texts) >= max_texts:
                            print(f"Loaded {len(texts)} texts (max_texts limit reached)")
                            return texts
                        current = []
                        in_text = False
                    else:
                        current.append(line.rstrip('\n'))
        
        print(f"Loaded {len(texts)} texts from dump")
        return texts

    @classmethod
    def load_test_data(cls, fname):
        """
        Load test data: one string per line (UTF-8). Each string is the context
        for which the model must predict the next character (3 guesses).
        """
        data = []
        with open(fname, encoding='utf-8') as f:
            for line in f:
                inp = line.rstrip('\n')  # strip newline only, preserve rest
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def build_vocabulary(self, text_sequences, vocab_size=500, min_frequency=10):
        """
        Build character vocabulary from text sequences
        
        Args:
            text_sequences: list of strings
            vocab_size: maximum vocabulary size
            min_frequency: minimum character frequency to include
        
        Returns:
            char2idx, idx2char dictionaries
        """
        from collections import Counter
        
        print("Building vocabulary...")
        # Count character frequencies
        combined_text = '\n'.join(text_sequences)
        char_counts = Counter(combined_text)
        
        # Sort by frequency
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Take top characters by frequency
        top_chars = [char for char, count in sorted_chars 
                    if count >= min_frequency][:vocab_size]
        
        # Always include essential characters
        essential_chars = [' ', '\n', '.', ',', '!', '?', '-', ':', ';', '"', "'", '(', ')', '[', ']']
        for char in essential_chars:
            if char not in top_chars:
                top_chars.append(char)
        
        # Create mappings
        char2idx = {'<PAD>': 0, '<UNK>': 1}
        idx2char = {0: '<PAD>', 1: '<UNK>'}
        
        for i, char in enumerate(top_chars, start=2):
            char2idx[char] = i
            idx2char[i] = char
        
        print(f"Vocabulary size: {len(char2idx)}")
        print(f"Most common chars: {top_chars[:50]}")
        
        return char2idx, idx2char

    def calculate_perplexity(self, dataloader, criterion):
        """
        Calculate perplexity on a dataset
        From N-gram LM lecture: Perplexity = 2^(cross-entropy) = exp(loss)
        Lower perplexity = better model
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                if self.use_transformer:
                    output = self.model(x)
                else:
                    output, _ = self.model(x)
                
                loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)  # Perplexity = e^(cross-entropy)
        return perplexity, avg_loss
    
    def run_train(self, data, work_dir):
        """
        Train the model with modern techniques from lectures
        
        Improvements:
        - Transformer architecture (Transformers lecture)
        - Perplexity evaluation (N-gram LM lecture)
        - Gradient clipping (Neural Networks lecture)
        - Learning rate scheduling (Neural Networks lecture)
        - Validation split for monitoring
        
        Args:
            data: list of text sequences (strings)
            work_dir: directory to save checkpoints
        """
        text_sequences = data
        
        if len(text_sequences) == 0:
            print("No training data available. Creating dummy model.")
            self.char2idx = {'<PAD>': 0, '<UNK>': 1, ' ': 2, 'e': 3, 't': 4}
            self.idx2char = {0: '<PAD>', 1: '<UNK>', 2: ' ', 3: 'e', 4: 't'}
            if self.use_transformer:
                self.model = CharTransformer(len(self.char2idx)).to(self.device)
            else:
                self.model = CharLSTM(len(self.char2idx)).to(self.device)
            return
        
        # Build vocabulary from training data
        print("Building vocabulary...")
        self.char2idx, self.idx2char = self.build_vocabulary(text_sequences, vocab_size=500)
        
        # Initialize model
        vocab_size = len(self.char2idx)
        print(f"Initializing model with vocab size {vocab_size}...")
        if self.use_transformer:
            self.model = CharTransformer(
                vocab_size=vocab_size,
                d_model=256,
                num_heads=8,
                num_layers=4,
                d_ff=1024,
                dropout=0.1
            ).to(self.device)
            print("Using Transformer architecture")
        else:
            self.model = CharLSTM(vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2).to(self.device)
            print("Using LSTM architecture")
        
        # Split data into train/validation (90/10 split)
        split_idx = int(0.9 * len(text_sequences))
        train_sequences = text_sequences[:split_idx]
        val_sequences = text_sequences[split_idx:]
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = CharDataset(train_sequences, self.char2idx, sequence_length=100)
        val_dataset = CharDataset(val_sequences, self.char2idx, sequence_length=100) if len(val_sequences) > 0 else None
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0) if val_dataset else None
        
        print(f"Training sequences: {len(train_dataset)}")
        if val_loader:
            print(f"Validation sequences: {len(val_dataset)}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)  # AdamW for better regularization
        
        # Learning rate scheduler (from Neural Networks lecture)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        
        # Training loop
        num_epochs = 10
        best_val_perplexity = float('inf')
        patience_counter = 0
        max_patience = 3
        
        print("\nStarting training...")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                
                if self.use_transformer:
                    output = self.model(x)
                else:
                    output, _ = self.model(x)
                
                # Reshape for loss calculation
                loss = criterion(output.view(-1, vocab_size), y.view(-1))
                loss.backward()
                
                # Gradient clipping (from Neural Networks lecture - prevents exploding gradients)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
            
            avg_train_loss = total_loss / max(num_batches, 1)
            train_perplexity = math.exp(avg_train_loss)
            
            # Validation phase
            if val_loader:
                val_perplexity, avg_val_loss = self.calculate_perplexity(val_loader, criterion)
                
                print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
                print(f'  Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}')
                print(f'  Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}')
                print(f'  Perplexity interpretation: Model is choosing from ~{val_perplexity:.0f} characters on average')
                
                # Learning rate scheduling based on validation perplexity
                scheduler.step(val_perplexity)
                
                # Early stopping
                if val_perplexity < best_val_perplexity:
                    best_val_perplexity = val_perplexity
                    patience_counter = 0
                    # Save best model
                    checkpoint_path = os.path.join(work_dir, 'best_model.checkpoint')
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'char2idx': self.char2idx,
                        'idx2char': self.idx2char,
                        'vocab_size': vocab_size,
                        'use_transformer': self.use_transformer
                    }, checkpoint_path)
                    print(f'  ✓ New best model saved (perplexity: {best_val_perplexity:.2f})')
                else:
                    patience_counter += 1
                    print(f'  No improvement. Patience: {patience_counter}/{max_patience}')
                
                if patience_counter >= max_patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    break
            else:
                print(f'\nEpoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, '
                      f'Train Perplexity: {train_perplexity:.2f}')
            
            print("=" * 80)
        
        print("\nTraining completed!")
        if val_loader:
            print(f"Best validation perplexity: {best_val_perplexity:.2f}")
            # Load best model
            checkpoint_path = os.path.join(work_dir, 'best_model.checkpoint')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded best model for final predictions")

    def run_pred(self, data, temperature=1.0):
        """
        Make predictions for test data
        
        Temperature sampling from Neural Networks lecture:
        - temperature < 1.0: more conservative (sharper distribution)
        - temperature = 1.0: use raw probabilities
        - temperature > 1.0: more diverse (flatter distribution)
        
        Args:
            data: list of input strings
            temperature: sampling temperature for diversity
        
        Returns:
            list of 3-character prediction strings
        """
        if self.model is None:
            # Fallback to random if model not loaded
            preds = []
            all_chars = string.ascii_letters
            for inp in data:
                top_guesses = [random.choice(all_chars) for _ in range(3)]
                preds.append(''.join(top_guesses))
            return preds
        
        self.model.eval()
        preds = []
        
        with torch.no_grad():
            for inp in data:
                # Handle empty input
                if len(inp) == 0:
                    # Predict common starting characters based on training data
                    preds.append(' et')
                    continue
                
                # Convert input string to indices
                input_indices = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in inp]
                
                # Limit sequence length to avoid issues
                max_context = 256
                if len(input_indices) > max_context:
                    input_indices = input_indices[-max_context:]
                
                # Prepare input tensor
                x = torch.tensor([input_indices], dtype=torch.long).to(self.device)
                
                # Get predictions
                if self.use_transformer:
                    output = self.model(x)  # Transformer returns logits directly
                else:
                    output, _ = self.model(x)  # LSTM returns output and hidden state
                
                # Get logits for the last character's prediction
                last_logits = output[0, -1, :]  # (vocab_size,)
                
                # Apply temperature scaling
                scaled_logits = last_logits / temperature
                
                # Get probabilities with softmax
                probs = F.softmax(scaled_logits, dim=0)
                
                # Get top 3 characters
                top3_indices = torch.topk(probs, k=min(10, len(probs))).indices.cpu().numpy()
                
                # Filter out special tokens and get top 3
                top3_chars = []
                for idx in top3_indices:
                    char = self.idx2char.get(int(idx), None)
                    if char and char not in ['<PAD>', '<UNK>']:
                        top3_chars.append(char)
                        if len(top3_chars) == 3:
                            break
                
                # Ensure we have 3 characters (fallback to common characters)
                common_fallbacks = [' ', 'e', 't', 'a', 'o', 'i', 'n']
                while len(top3_chars) < 3:
                    for fallback in common_fallbacks:
                        if fallback not in top3_chars:
                            top3_chars.append(fallback)
                            break
                    if len(top3_chars) < 3:
                        top3_chars.append(' ')
                
                preds.append(''.join(top3_chars[:3]))
        
        return preds

    def save(self, work_dir):
        """Save model checkpoint with architecture info"""
        if self.model is None:
            # Save dummy checkpoint if no model
            with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
                f.write('dummy save')
            return
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'char2idx': self.char2idx,
            'idx2char': self.idx2char,
            'vocab_size': len(self.char2idx),
            'use_transformer': self.use_transformer
        }
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        torch.save(checkpoint, checkpoint_path)
        print(f'Model saved to {checkpoint_path}')
        print(f'Architecture: {"Transformer" if self.use_transformer else "LSTM"}')

    @classmethod
    def load(cls, work_dir):
        """Load model checkpoint with architecture detection"""
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        
        # Check for best model first
        best_checkpoint_path = os.path.join(work_dir, 'best_model.checkpoint')
        if os.path.exists(best_checkpoint_path):
            checkpoint_path = best_checkpoint_path
            print(f'Loading best model from {checkpoint_path}...')
        
        # Check if it's a dummy save
        try:
            with open(checkpoint_path, 'rt') as f:
                content = f.read()
                if content == 'dummy save':
                    print("Loading dummy model")
                    return MyModel()
        except:
            pass
        
        # Load PyTorch checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Loading checkpoint from {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Detect architecture (default to LSTM for backward compatibility)
        use_transformer = checkpoint.get('use_transformer', False)
        
        # Create model instance
        model_instance = cls(
            vocab_size=checkpoint['vocab_size'],
            char2idx=checkpoint['char2idx'],
            idx2char=checkpoint['idx2char'],
            use_transformer=use_transformer
        )
        
        # Load model weights
        model_instance.model.load_state_dict(checkpoint['model_state_dict'])
        model_instance.model.eval()
        
        print(f'Model loaded successfully')
        print(f'Architecture: {"Transformer" if use_transformer else "LSTM"}')
        print(f'Vocabulary size: {checkpoint["vocab_size"]}')
        return model_instance


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))