"""
transformer_toy_mt.py
- Minimal Transformer encoder-decoder implemented from primitives
- Simple toy dataset generator included (small parallel corpus)
- Training loop, artifact saving to runs/mt/
Run:
 python code/transformer_toy_mt.py --epochs 40 --batch-size 64 --d_model 128
"""
import os
import math
import random
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utilities: tokenization & dataset
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

def build_vocab(sentences, min_freq=1):
    freq = {}
    for s in sentences:
        for tok in s.split():
            freq[tok] = freq.get(tok,0)+1
    idx2tok = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    tok2idx = {t:i for i,t in enumerate(idx2tok)}
    for tok,cnt in freq.items():
        if cnt>=min_freq:
            tok2idx.setdefault(tok, len(tok2idx))
            idx2tok.append(tok)
    return tok2idx, idx2tok

def encode_sentence(s, tok2idx, add_sos_eos=True):
    toks = s.split()
    ids = [tok2idx.get(t, tok2idx[UNK_TOKEN]) for t in toks]
    if add_sos_eos:
        return [tok2idx[SOS_TOKEN]] + ids + [tok2idx[EOS_TOKEN]]
    else:
        return ids

def pad_batch(batch, pad_idx):
    maxlen = max(len(x) for x in batch)
    arr = np.full((len(batch), maxlen), pad_idx, dtype=np.int64)
    for i,x in enumerate(batch):
        arr[i,:len(x)] = x
    return arr

def generate_toy_parallel(n_pairs=2000):
    # Simple synthetic "translation": reverse words + some token mapping to make it non-trivial
    src_sentences = []
    tgt_sentences = []
    base_vocab = ["i", "you", "he", "she", "we", "they", "like", "love", "hate", "eat", "drink", "apple", "banana", "rice", "rice", "today", "tomorrow", "yesterday", "big", "small", "fast", "slow", "a", "the"]
    for _ in range(n_pairs):
        L = random.randint(3,7)
        s = " ".join(random.choice(base_vocab) for _ in range(L))
        t = " ".join(reversed(s.split()))  # a simple target (reverse)
        src_sentences.append(s)
        tgt_sentences.append(t)
    return src_sentences, tgt_sentences

# Transformer primitive blocks
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        denom = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        def split(x):
            # x: B,T,D -> B, heads, T, d_k
            return x.view(B, -1, self.n_heads, self.d_k).transpose(1,2)
        q = split(self.w_q(q))
        k = split(self.w_k(k))
        v = split(self.w_v(v))
        # scaled dot-product
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)  # B,heads, Tq, Tk
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # B,heads, Tq, d_k
        out = out.transpose(1,2).contiguous().view(B, -1, self.n_heads*self.d_k)
        out = self.w_o(out)
        return out, attn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, src_mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        self_attn_out, self_attn_map = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self_attn_out)
        cross_out, cross_map = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + cross_out)
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        return x, self_attn_map, cross_map

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
    def forward(self, src, src_mask=None):
        x = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        attn_maps = []
        cross_maps = []
        for layer in self.layers:
            x, a, c = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            attn_maps.append(a)
            cross_maps.append(c)
        return x, attn_maps, cross_maps

class TransformerSimple(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, n_heads=4, d_ff=256, n_layers=2, max_len=100):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_heads, d_ff, n_layers, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, n_heads, d_ff, n_layers, max_len)
        self.out = nn.Linear(d_model, tgt_vocab)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        dec_out, attn_maps, cross_maps = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.out(dec_out)
        return logits, attn_maps, cross_maps

# Mask helpers (fixed)
def create_padding_mask(seq, pad_idx):
    # seq: B, T
    # Returns boolean mask: True = keep, False = mask out
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,T), bool

def create_look_ahead_mask(sz):
    # Lower triangular mask for causal decoding
    mask = torch.tril(torch.ones(sz, sz, dtype=torch.bool))  # (T,T), bool
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,sz,sz)

def combine_masks(pad_mask, lookahead_mask):
    if pad_mask is None:
        return lookahead_mask
    # both are bool now
    return pad_mask & lookahead_mask.to(pad_mask.device)


# BLEU (simple corpus BLEU implementation - unigram..4gram)
def ngram_counts(tokens, n):
    cnt = {}
    for i in range(len(tokens)-n+1):
        ng = tuple(tokens[i:i+n])
        cnt[ng] = cnt.get(ng,0)+1
    return cnt

def corpus_bleu(references_list, hypotheses_list, max_n=4):
    # references_list: list(list_of_tokens)
    # hypotheses_list: list(list_of_tokens)
    precisions = []
    for n in range(1, max_n+1):
        num=0; den=0
        for ref, hyp in zip(references_list, hypotheses_list):
            ref_cnt = ngram_counts(ref, n)
            hyp_cnt = ngram_counts(hyp, n)
            for ng, c in hyp_cnt.items():
                num += min(c, ref_cnt.get(ng,0))
            den += sum(hyp_cnt.values())
        precisions.append((num, den))
    # geometric mean
    p_log_sum = 0.0
    for num, den in precisions:
        if den==0: return 0.0
        if num==0:
            return 0.0
        p_log_sum += math.log(num/den)
    gm = math.exp(p_log_sum / max_n)
    # brevity penalty
    ref_len = sum(len(r) for r in references_list)
    hyp_len = sum(len(h) for h in hypotheses_list)
    bp = 1.0
    if hyp_len < ref_len:
        bp = math.exp(1 - ref_len/hyp_len) if hyp_len>0 else 0.0
    return 100 * bp * gm

# Training pipeline
def batchify(src_ids, tgt_ids, pad_idx, batch_size, shuffle=True):
    idxs = list(range(len(src_ids)))
    if shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        batch_idx = idxs[i:i+batch_size]
        src_b = pad_batch([src_ids[j] for j in batch_idx], pad_idx)
        tgt_b = pad_batch([tgt_ids[j] for j in batch_idx], pad_idx)
        yield torch.tensor(src_b, dtype=torch.long), torch.tensor(tgt_b, dtype=torch.long)

def train_epoch(model, data, opt, device, pad_idx):
    model.train()
    total_loss=0.0; total_tokens=0
    for src_b, tgt_b in batchify(data['src_ids'], data['tgt_ids'], pad_idx, data['batch_size']):
        src_b = src_b.to(device)
        tgt_b = tgt_b.to(device)
        tgt_in = tgt_b[:, :-1]
        tgt_out = tgt_b[:, 1:]
        src_mask = create_padding_mask(src_b, pad_idx).to(device)
        pad_mask = create_padding_mask(tgt_in, pad_idx).to(device)
        look = create_look_ahead_mask(tgt_in.size(1)).to(device)
        tgt_mask = combine_masks(pad_mask, look)
        opt.zero_grad()
        logits, _, _ = model(src_b, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
        logits_flat = logits.view(-1, logits.size(-1))
        tgt_out_flat = tgt_out.contiguous().view(-1)
        loss = F.cross_entropy(logits_flat, tgt_out_flat, ignore_index=pad_idx)
        loss.backward()
        opt.step()
        total_loss += loss.item() * (tgt_out_flat != pad_idx).sum().item()
        total_tokens += (tgt_out_flat != pad_idx).sum().item()
    return total_loss/total_tokens if total_tokens>0 else 0.0

def eval_epoch(model, data, device, pad_idx):
    model.eval()
    total_loss=0.0; total_tokens=0
    hyps=[]; refs=[]
    with torch.no_grad():
        for src_b, tgt_b in batchify(data['src_ids'], data['tgt_ids'], pad_idx, data['batch_size'], shuffle=False):
            src_b = src_b.to(device)
            tgt_b = tgt_b.to(device)
            tgt_in = tgt_b[:, :-1]
            tgt_out = tgt_b[:, 1:]
            src_mask = create_padding_mask(src_b, pad_idx).to(device)
            pad_mask = create_padding_mask(tgt_in, pad_idx).to(device)
            look = create_look_ahead_mask(tgt_in.size(1)).to(device)
            tgt_mask = combine_masks(pad_mask, look)
            logits, attn_maps, cross_maps = model(src_b, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
            logits_flat = logits.view(-1, logits.size(-1))
            tgt_out_flat = tgt_out.contiguous().view(-1)
            loss = F.cross_entropy(logits_flat, tgt_out_flat, ignore_index=pad_idx)
            total_loss += loss.item() * (tgt_out_flat!=pad_idx).sum().item()
            total_tokens += (tgt_out_flat!=pad_idx).sum().item()
            # greedy decode for BLEU
            for i in range(src_b.size(0)):
                # greedy decoding from model (teacher forcing input length)
                enc = model.encoder(src_b[i:i+1], src_mask=src_mask[i:i+1])
                # decode step-by-step
                cur = [data['tgt_tok2idx'][SOS_TOKEN]]
                for _s in range(50):
                    cur_t = torch.tensor([cur], dtype=torch.long).to(device)
                    pad_mask_dec = create_padding_mask(cur_t, pad_idx).to(device)
                    look = create_look_ahead_mask(cur_t.size(1)).to(device)
                    tgt_mask2 = combine_masks(pad_mask_dec, look)
                    logits_step, _, _ = model(src_b[i:i+1], cur_t, src_mask=src_mask[i:i+1], tgt_mask=tgt_mask2, memory_mask=src_mask[i:i+1])
                    token = logits_step[0,-1].argmax().item()
                    cur.append(token)
                    if token == data['tgt_tok2idx'][EOS_TOKEN]:
                        break
                hyp = [data['tgt_idx2tok'][id] for id in cur[1:-1]]  # remove sos eos
                ref = [data['tgt_idx2tok'][id] for id in tgt_b[i].cpu().numpy() if id not in (pad_idx, data['tgt_tok2idx'][SOS_TOKEN], data['tgt_tok2idx'][EOS_TOKEN])]
                hyps.append(hyp)
                refs.append(ref)
    bleu = corpus_bleu(refs, hyps)
    return total_loss/total_tokens if total_tokens>0 else 0.0, bleu, refs, hyps

# Visualize attention heatmap (first sample, first head, first layer)
def plot_attention(attn_map, out_path, title='attention'):
    # attn_map: (heads, Tq, Tk)
    h = attn_map[0]  # first head
    plt.figure(figsize=(6,6))
    plt.imshow(h, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('key (source positions)')
    plt.ylabel('query (target positions)')
    plt.savefig(out_path)
    plt.close()

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--d-model', type=int, default=128)
    args = parser.parse_args()
    out_dir = 'runs/mt'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset
    src_sents, tgt_sents = generate_toy_parallel(n_pairs=4000)
    # split
    split = int(0.9 * len(src_sents))
    src_train, src_val = src_sents[:split], src_sents[split:]
    tgt_train, tgt_val = tgt_sents[:split], tgt_sents[split:]
    # build vocabs
    src_tok2idx, src_idx2tok = build_vocab(src_train + src_val)
    tgt_tok2idx, tgt_idx2tok = build_vocab(tgt_train + tgt_val)
    # ensure special tokens
    for special in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
        if special not in src_tok2idx:
            src_tok2idx[special] = len(src_tok2idx); src_idx2tok.append(special)
        if special not in tgt_tok2idx:
            tgt_tok2idx[special] = len(tgt_tok2idx); tgt_idx2tok.append(special)

    # encode
    src_train_ids = [encode_sentence(s, src_tok2idx) for s in src_train]
    src_val_ids = [encode_sentence(s, src_tok2idx) for s in src_val]
    tgt_train_ids = [encode_sentence(s, tgt_tok2idx) for s in tgt_train]
    tgt_val_ids = [encode_sentence(s, tgt_tok2idx) for s in tgt_val]
    pad_idx = tgt_tok2idx[PAD_TOKEN]

    data = {
        'src_ids': src_train_ids,
        'tgt_ids': tgt_train_ids,
        'batch_size': args.batch_size,
        'tgt_tok2idx': tgt_tok2idx,
        'tgt_idx2tok': tgt_idx2tok
    }
    val_data = {
        'src_ids': src_val_ids,
        'tgt_ids': tgt_val_ids,
        'batch_size': args.batch_size,
        'tgt_tok2idx': tgt_tok2idx,
        'tgt_idx2tok': tgt_idx2tok
    }

    model = TransformerSimple(src_vocab=len(src_tok2idx), tgt_vocab=len(tgt_tok2idx),
                              d_model=args.d_model, n_heads=4, d_ff=4*args.d_model, n_layers=2, max_len=100).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {'train_loss':[], 'val_loss':[], 'val_bleu':[]}

    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, data, opt, device, pad_idx)
        val_loss, val_bleu, refs, hyps = eval_epoch(model, val_data, device, pad_idx)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_bleu'].append(val_bleu)
        print(f"Epoch {epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_bleu={val_bleu:.2f}")
        # save small artifacts
        torch.save(model.state_dict(), os.path.join(out_dir, 'latest.pth'))

    # save curves
    plt.figure(figsize=(6,4))
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss curves')
    plt.savefig(os.path.join(out_dir, 'curves_mt.png'))
    plt.close()

    # attention heatmap demo: run one sample through model and get cross attention from decoder layer 0 head 0
    model.eval()
    with torch.no_grad():
        src_b = torch.tensor(pad_batch(src_val_ids[:1], pad_idx), dtype=torch.long).to(device)
        tgt_b = torch.tensor(pad_batch(tgt_val_ids[:1], pad_idx), dtype=torch.long).to(device)
        tgt_in = tgt_b[:, :-1]
        src_mask = create_padding_mask(src_b, pad_idx).to(device)
        pad_mask = create_padding_mask(tgt_in, pad_idx).to(device)
        look = create_look_ahead_mask(tgt_in.size(1)).to(device)
        tgt_mask = combine_masks(pad_mask, look)
        logits, attn_maps, cross_maps = model(src_b, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
        # cross_maps: list of layer maps; shape per map: B, heads, Tq, Tk
        cross0 = cross_maps[0][0].cpu().numpy()  # first example, head=0
        # save attention heatmap
        plot_attention(cross0, os.path.join(out_dir, 'attention_layer0_head0.png'), title='cross-attention head0 layer0')

    # decodes table: pick 10 val samples and greedy decode
    decodes = []
    for i in range(min(10, len(src_val_ids))):
        src = torch.tensor(pad_batch([src_val_ids[i]], pad_idx), dtype=torch.long).to(device)
        enc = model.encoder(src, create_padding_mask(src, pad_idx).to(device))
        cur = [tgt_tok2idx[SOS_TOKEN]]
        for _ in range(50):
            cur_t = torch.tensor([cur], dtype=torch.long).to(device)
            pad_mask_dec = create_padding_mask(cur_t, pad_idx).to(device)
            look = create_look_ahead_mask(cur_t.size(1)).to(device)
            tgt_mask2 = combine_masks(pad_mask_dec, look)
            logits_step, _, _ = model(src, cur_t, src_mask=create_padding_mask(src, pad_idx).to(device), tgt_mask=tgt_mask2, memory_mask=create_padding_mask(src, pad_idx).to(device))
            tok = logits_step[0,-1].argmax().item()
            cur.append(tok)
            if tok == tgt_tok2idx[EOS_TOKEN]: break
        hyp = " ".join([tgt_idx2tok[id] for id in cur[1:-1]])
        ref = " ".join([tgt_idx2tok[id] for id in tgt_val_ids[i] if id not in (pad_idx, tgt_tok2idx[SOS_TOKEN], tgt_tok2idx[EOS_TOKEN])])
        decodes.append((src_val[i], ref, hyp))
    # save decodes table image (simple)
    fig, ax = plt.subplots(len(decodes),1, figsize=(8, 2*len(decodes)))
    if len(decodes)==1:
        ax=[ax]
    for i, (src, ref, hyp) in enumerate(decodes):
        ax[i].axis('off')
        ax[i].text(0,0.6, f"SRC: {src}")
        ax[i].text(0,0.4, f"REF: {ref}")
        ax[i].text(0,0.2, f"HYP: {hyp}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'decodes_table.png'))
    plt.close()

    # masks demo (visualize attention masks)
    m1 = create_padding_mask(torch.tensor(pad_batch(src_val_ids[:1], pad_idx)), pad_idx)[0,0]
    m2 = create_look_ahead_mask(10)[0,0]
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    plt.title('src padding mask')
    plt.imshow(m1.cpu().numpy(), aspect='auto', cmap='gray')

    plt.subplot(1,2,2)
    plt.title('tgt causal mask')
    plt.imshow(m2.cpu().numpy(), aspect='auto', cmap='gray')

    plt.savefig(os.path.join(out_dir, 'masks_demo.png'))
    plt.close()

    # BLEU report (recompute on val set)
    _, bleu, refs, hyps = eval_epoch(model, val_data, device, pad_idx)
    plt.figure(figsize=(4,2)); plt.text(0.1,0.5,f"Corpus BLEU: {bleu:.2f}", fontsize=14); plt.axis('off')
    plt.savefig(os.path.join(out_dir, 'bleu_report.png'))
    plt.close()
    print("Saved MT artifacts to", out_dir)

if __name__ == '__main__':
    main()
