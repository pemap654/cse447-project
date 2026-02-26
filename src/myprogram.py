#!/usr/bin/env python
import os
import random
import bz2
import pickle
from collections import Counter, defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
    Character n-gram language model with Kneser-Ney-inspired smoothing and
    linear interpolation across all n-gram orders for next-character prediction.

    Improvements over simple backoff (based on J&M Ch. 3 / CSE 447 lectures):
      1. Linear interpolation (slides 36-40): Instead of hard backoff that uses
         only the longest matching context, blend probability estimates from ALL
         matching contexts simultaneously. Longer contexts get higher weight.
         This is strictly more robust — it "never gives zero probability" and
         uses ALL available evidence rather than discarding lower-order info.
      2. Kneser-Ney continuation probability (slides 33-35): The base (unigram)
         distribution uses how many UNIQUE left-contexts each character appears
         in, rather than raw frequency. "Francisco" may be frequent but only
         follows "San" — poor backoff candidate. "e" follows many diverse
         contexts — great backoff candidate. This is the KN key insight.
      3. Absolute discounting: subtract a fixed D from each count before
         normalizing, freeing mass for the interpolated lower-order model.
      4. Storing (char, count) pairs per context (not just char tuples) enables
         real probability estimates for interpolation.
    """

    TRAIN_DUMP_PATHS = [
        'data/enwikiquote-2026-02-01-p1p303038.xml.bz2',
        'data/dump.xml.bz2',
        'dump.xml.bz2',
        '/job/data/dump.xml.bz2',
        '/job/data/enwikiquote-2026-02-01-p1p303038.xml.bz2',
    ]

    MAX_N = 8                    # Maximum context length for n-gram lookup
    MAX_TRAIN_CHARS = 20_000_000 # Cap corpus size to control memory/time
    DISCOUNT = 0.75              # Absolute discounting parameter (KN standard)
    TOP_K = 10                   # Store top-K chars per context (was 5)
    BASE_WEIGHT = 0.5            # Weight for KN base distribution in interpolation

    def __init__(self):
        # ctx_string -> tuple of (char, count) pairs (most common first, top-K)
        self.ngrams = {}
        # ctx_string -> total count for that context (sum of all char counts)
        self.ctx_totals = {}
        # char -> number of unique single-char left-contexts (KN continuation)
        self.cont_count = Counter()
        # total unique bigrams (normalizer for continuation probability)
        self.total_unique_bigrams = 0
        # raw unigram frequency (fallback if cont_count unavailable)
        self.char_freq = Counter()
        self.max_n = self.MAX_N
        # True only for freshly trained models with real count data.
        # Old-format checkpoints use rank-based proxies which don't faithfully
        # represent probabilities, so we fall back to hard backoff for those.
        self._has_real_counts = False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @classmethod
    def load_training_data(cls, max_texts=None):
        """Load text from bz2-compressed MediaWiki XML dump."""
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
                                print(f"Loaded {len(texts)} texts (limit reached)")
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
                            print(f"Loaded {len(texts)} texts (limit reached)")
                            return texts
                        current = []
                        in_text = False
                    else:
                        current.append(line.rstrip('\n'))

        print(f"Loaded {len(texts)} texts from dump")
        return texts

    @classmethod
    def load_test_data(cls, fname):
        """Load test data: one context string per line."""
        data = []
        with open(fname, encoding='utf-8') as f:
            for line in f:
                data.append(line.rstrip('\n'))
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run_train(self, data, work_dir):
        """
        Build character n-gram model with KN-style statistics.

        Stores:
          - ngrams[ctx]: top-K (char, count) pairs (sorted by frequency)
          - ctx_totals[ctx]: total count (denominator for probabilities)
          - cont_count[c]: # unique single-char left-contexts → KN continuation
          - total_unique_bigrams: normalizer for continuation probability
        """
        if not data:
            print("No training data available.")
            return

        print(f"Preparing corpus from {len(data)} texts...")
        combined = '\n'.join(data).lower()
        if len(combined) > self.MAX_TRAIN_CHARS:
            combined = combined[:self.MAX_TRAIN_CHARS]
            print(f"Corpus capped at {self.MAX_TRAIN_CHARS:,} characters")
        n = len(combined)
        print(f"Corpus size: {n:,} characters")

        # Build n-gram counts
        ngram_counts = defaultdict(Counter)
        skip = set('\r\t\x00\x01\x02\x03')

        print(f"Building n-gram counts (up to {self.max_n}-grams)...")
        for ctx_len in range(1, self.max_n + 1):
            for i in range(ctx_len, n):
                nchar = combined[i]
                if nchar in skip:
                    continue
                ctx = combined[i - ctx_len:i]
                ngram_counts[ctx][nchar] += 1
            print(f"  ctx_len={ctx_len}: {len(ngram_counts):,} unique contexts so far")

        # Raw unigram frequency (backup fallback)
        for ch in combined:
            if ch not in skip:
                self.char_freq[ch] += 1

        # KN continuation counts:
        # cont_count[c] = # unique single-char left-contexts preceding c
        # This captures context diversity (J&M / lecture slide 34-35).
        # Characters like 'e' follow many diverse contexts → high cont_count.
        # Characters like niche letter combos follow few → low cont_count.
        print("Building KN continuation counts...")
        self.cont_count = Counter()
        for ctx, counter in ngram_counts.items():
            if len(ctx) == 1:  # bigram: single-char left context
                for c in counter:
                    self.cont_count[c] += 1
        self.total_unique_bigrams = sum(self.cont_count.values())
        print(f"  {len(self.cont_count)} chars, {self.total_unique_bigrams:,} unique bigrams")

        # Prune: keep top-K printable chars per context, store with counts
        print(f"Pruning model to top-{self.TOP_K} chars per context...")
        self.ngrams = {}
        self.ctx_totals = {}
        for ctx, counter in ngram_counts.items():
            top = [(c, cnt) for c, cnt in counter.most_common(self.TOP_K + 5)
                   if ord(c) >= 32][:self.TOP_K]
            if top:
                self.ngrams[ctx] = tuple(top)
                self.ctx_totals[ctx] = sum(counter.values())

        self._has_real_counts = True
        print(f"Model ready: {len(self.ngrams):,} n-gram contexts")
        print(f"Top chars: {[c for c, _ in self.char_freq.most_common(10)]}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def run_pred(self, data):
        """
        Predict next character.

        Uses KN-interpolated scoring when the model was trained with the new
        code (real counts stored). Falls back to hard backoff for old-format
        checkpoints where only rank proxies are available — interpolating rank
        proxies can dilute high-order evidence and hurt accuracy.

        Interpolation strategy (J&M Ch.3 / lecture slides 36-40):
          Score each candidate by blending discounted probabilities from ALL
          matching n-gram orders, weighting longer contexts more heavily.
          This is more robust than hard backoff and never gives zero prob.

        Hard-backoff strategy (old format):
          Use only the longest matching context's top candidates, with KN
          continuation probability as the base fallback.
        """
        if self._has_real_counts:
            return self._run_pred_interpolated(data)
        else:
            return self._run_pred_backoff(data)

    def _build_kn_base(self):
        """Return (kn_base_dict, kn_sorted_list) for use in prediction."""
        if self.total_unique_bigrams > 0:
            kn_base = {c: cnt / self.total_unique_bigrams
                       for c, cnt in self.cont_count.items()
                       if c != '\n'}
        else:
            total_chars = sum(self.char_freq.values()) or 1
            kn_base = {c: cnt / total_chars
                       for c, cnt in self.char_freq.items()
                       if c != '\n'}
        kn_sorted = sorted(kn_base, key=kn_base.get, reverse=True)
        return kn_base, kn_sorted

    def _run_pred_interpolated(self, data):
        """
        Interpolated KN prediction for freshly trained models with real counts.
        Blends all n-gram orders; longer contexts get higher weight.
        """
        D = self.DISCOUNT
        BASE_W = self.BASE_WEIGHT
        kn_base, kn_sorted = self._build_kn_base()

        preds = []
        for inp in data:
            ctx = inp.lower()
            scores = {}

            for ctx_len in range(1, min(len(ctx), self.max_n) + 1):
                key = ctx[-ctx_len:]
                if key not in self.ngrams:
                    continue
                total = self.ctx_totals[key]
                w = float(ctx_len)
                for c, count in self.ngrams[key]:
                    p = max(count - D, 0.0) / total
                    scores[c] = scores.get(c, 0.0) + w * p

            for c, p_kn in kn_base.items():
                scores[c] = scores.get(c, 0.0) + BASE_W * p_kn

            top3 = sorted(scores, key=scores.__getitem__, reverse=True)[:3]
            for c in kn_sorted:
                if len(top3) >= 3:
                    break
                if c not in top3:
                    top3.append(c)

            preds.append(''.join(top3[:3]))
        return preds

    def _run_pred_backoff(self, data):
        """
        Hard backoff prediction for old-format checkpoints (rank-proxy counts).
        Uses only the longest matching context; KN continuation as base fallback.
        """
        _, kn_sorted = self._build_kn_base()

        preds = []
        for inp in data:
            ctx = inp.lower()
            top3 = []

            for ctx_len in range(min(len(ctx), self.max_n), 0, -1):
                key = ctx[-ctx_len:]
                if key in self.ngrams:
                    top3 = [c for c, _ in self.ngrams[key]][:3]
                    break

            for c in kn_sorted:
                if len(top3) >= 3:
                    break
                if c not in top3:
                    top3.append(c)

            preds.append(''.join(top3[:3]))
        return preds

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, work_dir):
        """Save model as a bz2-compressed pickle."""
        os.makedirs(work_dir, exist_ok=True)
        path = os.path.join(work_dir, 'model.checkpoint')
        checkpoint = {
            'ngrams': self.ngrams,
            'ctx_totals': self.ctx_totals,
            'cont_count': self.cont_count,
            'total_unique_bigrams': self.total_unique_bigrams,
            'char_freq': self.char_freq,
            'max_n': self.max_n,
        }
        print(f"Saving model to {path} ...")
        with bz2.open(path, 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / 1e6
        print(f"Saved ({size_mb:.1f} MB compressed)")

    @classmethod
    def load(cls, work_dir):
        """Load model from bz2-compressed pickle checkpoint."""
        model = cls()
        path = os.path.join(work_dir, 'model.checkpoint')
        print(f"Loading model from {path} ...")
        try:
            with bz2.open(path, 'rb') as f:
                chk = pickle.load(f)
            model.char_freq = chk['char_freq']
            model.max_n = chk.get('max_n', 8)
            model.cont_count = chk.get('cont_count', Counter())
            model.total_unique_bigrams = chk.get('total_unique_bigrams', 0)

            raw_ngrams = chk['ngrams']
            if 'ctx_totals' in chk:
                # New format: already stores (char, count) pairs with real counts
                model.ngrams = raw_ngrams
                model.ctx_totals = chk['ctx_totals']
                model._has_real_counts = True
            else:
                # Old format: stores plain char tuples — convert using rank-based counts
                print("Converting old checkpoint format to new (char, count) format...")
                model.ngrams = {}
                model.ctx_totals = {}
                for ctx, chars in raw_ngrams.items():
                    n = len(chars)
                    pairs = tuple((c, n - i) for i, c in enumerate(chars))
                    model.ngrams[ctx] = pairs
                    model.ctx_totals[ctx] = sum(n - i for i in range(n))
                print("Conversion done.")

            print(f"Model loaded: {len(model.ngrams):,} n-gram contexts")
        except Exception as e:
            print(f"Warning: could not load model ({e}). Using unigram fallback.")
        return model


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data',
                        default='example/input.txt')
    parser.add_argument('--test_output', help='path to write predictions',
                        default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

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
        assert len(pred) == len(test_data), \
            'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
