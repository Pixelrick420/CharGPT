"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text):
    """Return a sorted list of unique characters in text."""
    return sorted(list(set(text)))

# Step 2 - build_stoi
def build_stoi(vocab):
    out = dict()
    for index, char in enumerate(vocab):
        out[char] = index
    
    return out

# Step 3 - build_itos
def build_itos(vocab):
    out = dict()
    for index, char in enumerate(vocab):
        out[index] = char
    
    return out

# Step 4 - encode_char
def encode_char(ch, stoi):
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text, stoi):
    return [
        stoi[char] for char in text
    ]

# Step 6 - decode_int
def decode_int(token_id, itos):
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids, itos):
    return ''.join([
        itos[token_id] for token_id in ids
    ])

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values):
    return np.array(values)

# Step 9 - get_array_shape
import numpy as np

def get_array_shape(arr):
    return arr.shape

# Step 10 - get_array_dtype
import numpy as np

def get_array_dtype(arr):
    return arr.dtype

# Step 11 - make_2d_zeros
import numpy as np

def make_2d_zeros(rows, cols):
    return np.zeros((rows, cols), dtype=np.int64)

# Step 12 - make_2d_random
import numpy as np

def make_2d_random(rows, cols, seed):
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols))

# Step 13 - index_element
def index_element(arr, i, j):
    return arr[i][j]

# Step 14 - slice_row
import numpy as np

def slice_row(arr, i):
    return arr[i]

# Step 15 - slice_column
import numpy as np

def slice_column(arr, j):
    return arr[:,j]

# Step 16 - slice_subblock
import numpy as np

def slice_subblock(arr, r0, r1, c0, c1):
    return arr[r0:r1, c0:c1]

# Step 17 - elementwise_add
import numpy as np

def elementwise_add(a, b):
    return a + b

# Step 18 - elementwise_multiply
import numpy as np

def elementwise_multiply(a, b):
    return a * b

# Step 19 - scalar_broadcast_add
import numpy as np

def scalar_broadcast_add(arr, scalar):
    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
import numpy as np

def vector_matrix_broadcast_add(matrix, vector):
    return matrix + vector

# Step 21 - array_exp
import numpy as np

def array_exp(arr):
    return np.exp(arr)

# Step 22 - array_log
import numpy as np

def array_log(arr):
    return np.log(arr)

# Step 23 - sum_all
import numpy as np

def sum_all(arr):
    return np.sum(arr)

# Step 24 - sum_axis0
import numpy as np

def sum_axis0(arr):
    return np.sum(arr, axis=0)

# Step 25 - sum_axis1
import numpy as np

def sum_axis1(arr):
    return np.sum(arr, axis=1)

# Step 26 - max_along_axis
import numpy as np

def max_along_axis(arr, axis):
    return np.max(arr, axis=axis)

# Step 27 - matmul
import numpy as np

def matmul(a, b):
    return a.dot(b)

# Step 28 - transpose_matrix
def transpose_matrix(arr):
    return np.transpose(arr)

# Step 29 - sum_keepdims
import numpy as np

def sum_keepdims(arr, axis):
    return np.sum(arr, axis=axis, keepdims=True)

# Step 30 - naive_softmax_1d
import numpy as np

def naive_softmax_1d(logits):
    exponents = np.exp(logits)
    return exponents / (np.sum(exponents))

# Step 31 - softmax_overflow_demo
def softmax_overflow_demo(large_value):
    out = dict()
    out['naive_exp'] = np.exp(large_value)
    out['overflowed'] = out['naive_exp'] == np.inf
    return out

# Step 32 - stable_softmax_1d
import numpy as np

def stable_softmax_2d_rowwise(logits):
    max_vals = max_along_axis(logits, axis=1, keepdims=True)
    shifted_logits = logits - max_vals
    exp_logits = array_exp(shifted_logits)
    sum_exp = sum_keepdims(exp_logits, axis=1)
    
    return exp_logits / sum_exp

# Step 33 - stable_softmax_2d_rowwise
import numpy as np

def stable_softmax_2d_rowwise(logits):
    max_vals = max_along_axis(logits, axis=1)
    shifted_logits = logits - max_vals[:, None]
    exp_logits = array_exp(shifted_logits)
    sum_exp = sum_keepdims(exp_logits, axis=1)
    return exp_logits / sum_exp

# Step 34 - read_text_file
def read_text_file(text_blob):
    if not isinstance(text_blob, str):
        raise TypeError()
    elif text_blob:
        return text_blob
    else:
        raise ValueError()

# Step 35 - encode_corpus_to_int_array
def encode_corpus_to_int_array(text, stoi):
    return np.array(
        [stoi[char] for char in text],
        dtype = np.int64
    )

# Step 36 - pick_split_point
def pick_split_point(n, train_frac):
    return int(n * train_frac)

# Step 37 - slice_train_and_val
def slice_train_and_val(data, split_idx):
    return (data[:split_idx], data[split_idx:])

# Step 38 - pick_block_size
def pick_block_size(default_size):
    return max(default_size, 1)

# Step 39 - slice_x_at_offset
import numpy as np

def slice_x_at_offset(data, i, block_size):
    return data[i : i + block_size]

# Step 40 - slice_y_at_offset
import numpy as np

def slice_y_at_offset(data, i, block_size):
    return data[i+1 : i+1+block_size]

# Step 41 - sample_random_batch_offsets
def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    return rng.integers(0, data_len - block_size, size=batch_size)

# Step 42 - stack_x_batch
import numpy as np

def stack_x_batch(data, offsets, block_size):
    return np.vstack([
        data[offset: offset + block_size] for offset in offsets
    ])

# Step 43 - stack_y_batch
import numpy as np

def stack_y_batch(data, offsets, block_size):
        return np.vstack([
        data[offset + 1: offset + 1 + block_size] for offset in offsets
    ])

# Step 44 - get_batch
def get_batch(data, block_size, batch_size, rng):
    offsets = sample_random_batch_offsets(len(data), block_size, batch_size, rng)
    return (
        stack_x_batch(data, offsets, block_size),
        stack_y_batch(data, offsets, block_size)
    )

# Step 45 - allocate_count_matrix
import numpy as np

def allocate_count_matrix(vocab_size):
    return np.zeros((vocab_size, vocab_size), dtype = np.int64)

# Step 46 - loop_fill_counts
import numpy as np

def loop_fill_counts(n_matrix, data):
    n = len(data)
    for i in range(1, n):
        n_matrix[data[i - 1]][data[i]] += 1
    
    return n_matrix

# Step 47 - vectorize_counts_add_at
import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    count_matrix = allocate_count_matrix(vocab_size)
    np.add.at(count_matrix, (data[:-1], data[1:]), 1)
    return count_matrix

# Step 48 - add_one_smoothing
import numpy as np

def add_one_smoothing(n_matrix):
    return n_matrix + 1

# Step 49 - row_sums_of_counts
def row_sums_of_counts(n_matrix):
    return sum_keepdims(n_matrix, 1)

# Step 50 - normalize_counts_to_probs
def normalize_counts_to_probs(n_matrix):
    return n_matrix / row_sums_of_counts(n_matrix)

# Step 51 - sample_next_token
def sample_next_token(p_matrix, current_id, rng):
    probabilities = p_matrix[current_id]
    V = len(probabilities)
    
    return rng.choice(V, p=probabilities)

# Step 52 - generate_sequence
def generate_sequence(p_matrix, start_id, length, rng):
    sequence = np.zeros(length, dtype=int)
    sequence[0] = start_id
    current_id = start_id
    
    for i in range(1, length):
        current_id = sample_next_token(p_matrix, current_id, rng)
        sequence[i] = current_id
        
    return sequence

# Step 53 - decode_generated_sequence
def decode_generated_sequence(ids, itos):
    return ''.join([itos[token_id] for token_id in ids])

# Step 54 - log_prob_of_pair
def log_prob_of_pair(p_matrix, current_id, next_id):
    return np.log(p_matrix[current_id][next_id])

# Step 55 - sum_negative_log_probs
def sum_negative_log_probs(p_matrix, data):
    n = len(data)
    return np.sum([-log_prob_of_pair(p_matrix, data[i - 1], data[i]) for i in range(1, n)])

# Step 56 - average_nll
def average_nll(p_matrix, data):
    total_nll = sum_negative_log_probs(p_matrix, data)
    return float(total_nll / (len(data) - 1))

# Step 57 - initialize_w_random
import numpy as np

def initialize_w_random(vocab_size, rng):
    return rng.standard_normal(size=(vocab_size, vocab_size))

# Step 58 - scale_w_small
import numpy as np

def scale_w_small(w_matrix, scale):
    return w_matrix * scale

# Step 59 - one_hot_encode_batch
import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    N = len(ids)
    onehot = np.zeros((N, vocab_size), dtype=np.float64)
    rows = np.arange(N)
    onehot[rows, ids] = 1.0
    return onehot

# Step 60 - forward_logits_onehot
def forward_logits_onehot(onehot, w_matrix):
    return matmul(onehot, w_matrix)

# Step 61 - observe_lookup_equivalence
import numpy as np

def observe_lookup_equivalence(w, ids):
    vocab_size = w.shape[0]
    onehot = one_hot_encode_batch(ids, vocab_size)
    
    return {
        'onehot_result': matmul(onehot, w),
        'index_result': w[ids]
    }

# Step 62 - forward_logits_lookup
def forward_logits_lookup(w, ids):
    return w[ids]

# Step 63 - logits_to_probs_rowwise
def logits_to_probs_rowwise(logits):
    return stable_softmax_2d_rowwise(logits)

# Step 64 - gather_correct_token_probs
def gather_correct_token_probs(probs, targets):
    return np.array([
        probs[i, targets[i]] for i in range(len(targets))
    ])

# Step 65 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, targets):
    correct_probs = gather_correct_token_probs(probs, targets)
    log_probs = array_log(correct_probs)
    return -np.mean(log_probs)

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper():
    return """Derivation of the gradient of mean cross-entropy loss with respect to the logits:

1. For a single example i, let z_i be the logits vector and y_i be the correct target class.
2. The probabilities are given by the softmax function: p_i = softmax(z_i).
3. The cross-entropy loss for this single example is L_i = -log(p_{i, y_i}).
4. Taking the derivative of L_i with respect to the j-th logit z_{i,j} yields a well-known result:
   - If j == y_i: d(L_i)/d(z_{i,j}) = p_{i,j} - 1
   - If j != y_i: d(L_i)/d(z_{i,j}) = p_{i,j}
   In vector form, this is: d(L_i)/d(z_i) = p_i - onehot(y_i).
5. The total loss L is the mean over the batch of size B: L = (1/B) * sum_{i=1}^B L_i.
6. Therefore, the gradient of the mean loss with respect to the entire logits matrix is the average of the individual gradients.

Final formula:
dL/dlogits = (probs - onehot(targets)) / B
"""

# Step 67 - compute_dlogits
import numpy as np

def compute_dlogits(probs, targets):
    B = probs.shape[0]
    dlogits = probs.copy()
    dlogits[np.arange(B), targets] -= 1.0
    dlogits /= B
    return dlogits

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper():
    return """Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].\nShapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).\nChain rule: dL/dW = O.T @ dlogits, shape (V, D).\nSince O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.\nRow v of dW equals the sum of dlogits[b] over all b with ids[b] == v.\nImplementation: scatter-add dlogits rows into dW at indices ids."""

# Step 69 - compute_dw_scatter_add
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    _, V_out = dlogits.shape
    dW = np.zeros((vocab_size, V_out), dtype=dlogits.dtype)
    np.add.at(dW, ids, dlogits)
    return dW

# Step 70 - sgd_update_w
import numpy as np

def sgd_update_w(w, dw, learning_rate):
    return (w - learning_rate * dw)

# Step 71 - run_one_training_step
def run_one_training_step(w, ids, targets, learning_rate):
    logits = forward_logits_lookup(w, ids)
    probs = logits_to_probs_rowwise(logits)
    loss = cross_entropy_loss(probs, targets)
    
    dlogits = compute_dlogits(probs, targets)
    vocab_size = w.shape[0]
    dw = compute_dw_scatter_add(ids, dlogits, vocab_size)
    
    updated_w = sgd_update_w(w, dw, learning_rate)
    
    return {'w': updated_w, 'loss': loss}

# Step 72 - train_neural_bigram_loop
import numpy as np

def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    loss_history = []
    rng = np.random.default_rng(42)
    
    for step in range(num_steps):
        ids, targets = get_batch(data, block_size, batch_size, rng)
        
        result = run_one_training_step(w, ids.flatten(), targets.flatten(), learning_rate)
        w = result['w']
        
        if step % log_every == 0:
            loss_history.append(result['loss'])
            
    return {'w': w, 'loss_history': loss_history}

# Step 73 - sample_from_neural_bigram
import numpy as np

def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    sequence = [start_id]
    current_id = start_id
    
    for _ in range(num_tokens):
        logits = forward_logits_lookup(w, [current_id])
        probs = logits_to_probs_rowwise(logits)[0]
        current_id = np.random.choice(len(probs), p=probs)
        sequence.append(current_id)
        
    return decode_ids(sequence, itos)

# Step 74 - linear_forward
def linear_forward(x, w):
    return {
        'y' : matmul(x, w),
        'cache' : {
            'x' : x,
            'w' : w
        }
    }

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper():
        return "Y = X @ W\ndL/dX = dY @ W.T\nshapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper():
    return """Forward pass of the linear layer: Y = X @ W
Shapes: X is (N, D_in), W is (D_in, D_out), Y is (N, D_out), and dY is (N, D_out).
By the multivariate chain rule, the gradient dW must have the same shape as W (D_in, D_out).
To achieve this shape and sum the gradients across the batch dimension N, we multiply the transpose of X by dY.
Therefore, the gradient of the loss with respect to W is:
dL/dW = X.T @ dY"""

# Step 77 - linear_backward_dx
def linear_backward_dx(dy, cache):
    w = cache['w']
    return dy.dot(w.T)

# Step 78 - linear_backward_dw
def linear_backward_dw(dy, cache):
    x = cache['x']
    return x.T.dot(dy)

# Step 79 - bias_add_forward
def bias_add_forward(x, b):
    return {
        'y' : x + b,
        'cache' : {
            'b_shape' : b.shape
        }
    }

# Step 80 - bias_add_backward_db
def bias_add_backward_db(dy, cache):
    return np.sum(dy, axis=0)

# Step 81 - relu_forward
def relu_forward(x):
    return {
        'y' : np.maximum(x, 0),
        'cache' : {
            'x' : x
        }
    }

# Step 82 - relu_backward
def relu_backward(dy, cache):
    x = cache['x']
    return dy * (x > 0)

# Step 83 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(probs, targets):
    return compute_dlogits(probs, targets)

# Step 84 - layernorm_forward_mean
import numpy as np

def layernorm_forward_mean(x):
    return np.mean(x, axis=-1, keepdims=True)

# Step 85 - layernorm_forward_variance
import numpy as np

def layernorm_forward_variance(x, mean):
    return np.mean((x - mean) ** 2, axis=-1, keepdims=True)

# Step 86 - layernorm_forward_normalize
import numpy as np

def layernorm_forward_normalize(x, mean, var, eps):
    return (
        (x - mean) / np.sqrt(var + eps)
    )

# Step 87 - layernorm_forward_affine
def layernorm_forward_affine(x, gamma, beta, eps):
    mean = layernorm_forward_mean(x)
    var = layernorm_forward_variance(x, mean)
    x_hat = layernorm_forward_normalize(x, mean, var, eps)
    
    scaled = elementwise_multiply(x_hat, gamma)
    y = vector_matrix_broadcast_add(scaled, beta)
    
    return {
        'y': y,
        'cache': {
            'x': x,
            'x_hat': x_hat,
            'mean': mean,
            'var': var,
            'gamma': gamma,
            'eps': eps
        }
    }

# Step 88 - layernorm_backward_subtract_mean
import numpy as np

def layernorm_backward_subtract_mean(dy, cache):
    return dy - np.mean(dy, axis=-1, keepdims=True)

# Step 89 - layernorm_backward_divide_std
def layernorm_backward_divide_std(dy, cache):
    var = cache['var']
    eps = cache['eps']
    std = np.sqrt(var + eps)
    return dy / std

# Step 90 - layernorm_backward_full
def layernorm_backward_full(dy, cache):
    x_hat = cache['x_hat']
    gamma = cache['gamma']
    var = cache['var']
    eps = cache['eps']
    
    sum_axes = tuple(range(dy.ndim - 1))
    dgamma = np.sum(dy * x_hat, axis=sum_axes)
    dbeta = np.sum(dy, axis=sum_axes)
    
    dx_hat = dy * gamma
    std = np.sqrt(var + eps)
    mean_dx_hat = np.mean(dx_hat, axis=-1, keepdims=True)
    mean_dx_hat_x_hat = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
    
    dx = (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat) / std
    
    return {
        'dx': dx,
        'dgamma': dgamma,
        'dbeta': dbeta
    }

# Step 91 - layernorm_backward_implementation
import numpy as np

def layernorm_backward_implementation(d_out, cache):
    x_hat = cache['x_hat']
    gamma = cache['gamma']
    var = cache['var']
    eps = cache['eps']
    
    dgamma = np.sum(d_out * x_hat, axis=0)
    dbeta = np.sum(d_out, axis=0)
    dx_hat = d_out * gamma
    
    std = np.sqrt(var + eps)
    mean_dx_hat = np.mean(dx_hat, axis=-1, keepdims=True)
    mean_dx_hat_x_hat = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
    
    dx = (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat) / std
    
    return {
        'dx': dx,
        'dgamma': dgamma,
        'dbeta': dbeta
    }

# Step 92 - create_token_embedding
def create_token_embedding(vocab_size, d_model, scale=0.02):
    return np.random.randn(vocab_size, d_model) * scale

# Step 93 - token_embedding_forward
def token_embedding_forward(token_ids, embedding_matrix):
    out = embedding_matrix[token_ids]
    cache = {
        'token_ids': token_ids,
        'vocab_size': embedding_matrix.shape[0]
    }
    return out, cache

# Step 94 - token_embedding_backward
def token_embedding_backward(d_out, cache):
    token_ids = cache['token_ids']
    vocab_size = cache['vocab_size']
    d_model = d_out.shape[-1]
    
    dE = np.zeros((vocab_size, d_model), dtype=d_out.dtype)
    np.add.at(dE, token_ids, d_out)
    
    return dE

# Step 95 - create_positional_embedding
def create_positional_embedding(block_size, d_model, scale=0.02):
    P = make_2d_random(block_size, d_model, seed=None)
    return scale_w_small(P, scale)

# Step 96 - slice_positional_embedding
import numpy as np

def slice_positional_embedding(positional_matrix, seq_len):
    return positional_matrix[:seq_len, :]

# Step 97 - add_token_and_positional_embeddings
def add_token_and_positional_embeddings(tok_emb, pos_emb):
    return tok_emb + pos_emb

# Step 98 - embedding_sum_backward
def embedding_sum_backward(d_out):
    return {
        'd_token_emb': d_out,
        'd_pos_emb': sum_axis0(d_out)
    }

# Step 99 - create_qkv_projections
def create_qkv_projections(d_model, d_head, scale=0.02):
    return {
        'Wq': scale_w_small(make_2d_random(d_model, d_head, 0), scale),
        'Wk': scale_w_small(make_2d_random(d_model, d_head, 1), scale),
        'Wv': scale_w_small(make_2d_random(d_model, d_head, 2), scale)
    }

# Step 100 - compute_query
def compute_query(x, w_q):
    return x @ w_q

# Step 101 - compute_key
def compute_key(x, w_k):
    return x @ w_k

# Step 102 - compute_value
def compute_value(x, w_v):
    return x @ w_v

# Step 103 - compute_attention_scores
def compute_attention_scores(q, k):
    return q @ k.swapaxes(1, 2)

# Step 104 - scale_attention_scores
def scale_attention_scores(scores, d_head):
    return scores / np.sqrt(d_head)

# Step 105 - build_causal_mask
def build_causal_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))

# Step 106 - apply_causal_mask
def apply_causal_mask(scaled_scores, causal_mask):
    return np.where(causal_mask, scaled_scores, -np.inf)

# Step 107 - softmax_attention_weights
def softmax_attention_weights(masked_scores):
    row_max = np.max(masked_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(masked_scores - row_max)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

# Step 108 - attention_weighted_values
import numpy as np

def attention_weighted_values(attn, v):
    return attn @ v

# Step 109 - apply_output_projection
import numpy as np

def apply_output_projection(attn_out, w_o):
    return attn_out @ w_o

# Step 110 - output_projection_backward
def output_projection_backward(d_proj, cache):
    attn_out = cache['attn_out']
    w_o = cache['w_o']
    
    d_attn_out = d_proj @ w_o.T
    
    attn_out_flat = attn_out.reshape(-1, attn_out.shape[-1])
    d_proj_flat = d_proj.reshape(-1, d_proj.shape[-1])
    dw_o = attn_out_flat.T @ d_proj_flat
    
    return {
        'd_attn_out': d_attn_out,
        'dw_o': dw_o
    }

# Step 111 - attention_value_backward
def attention_value_backward(d_attn_out, cache):
    attn = cache['attn']
    v = cache['v']
    
    d_attn = d_attn_out @ v.swapaxes(-1, -2)
    d_v = attn.swapaxes(-1, -2) @ d_attn_out
    
    return {
        'd_attn': d_attn,
        'd_v': d_v
    }

# Step 112 - masked_softmax_backward
def masked_softmax_backward(d_attn, cache):
    attn = cache['attn']
    causal_mask = cache['causal_mask']
    
    sum_term = np.sum(attn * d_attn, axis=-1, keepdims=True)
    
    d_scores = attn * (d_attn - sum_term)
    d_scores = np.where(causal_mask, d_scores, 0.0)
    
    return d_scores

# Step 113 - scale_scores_backward
def scale_scores_backward(d_scaled_scores, d_head):
    return d_scaled_scores / np.sqrt(d_head)

# Step 114 - qk_scores_backward
def qk_scores_backward(d_scores, cache):
    q = cache['q']
    k = cache['k']
    
    d_q = d_scores @ k
    d_k = d_scores.swapaxes(-1, -2) @ q
    
    return {
        'd_q': d_q,
        'd_k': d_k
    }

# Step 115 - qkv_projection_backward
def qkv_projection_backward(d_q, d_k, d_v, cache):
    x = cache['x']
    w_q = cache['w_q']
    w_k = cache['w_k']
    w_v = cache['w_v']
    
    dx_q = d_q @ w_q.T
    dx_k = d_k @ w_k.T
    dx_v = d_v @ w_v.T
    dx = dx_q + dx_k + dx_v
    
    x_flat = x.reshape(-1, x.shape[-1])
    d_q_flat = d_q.reshape(-1, d_q.shape[-1])
    d_k_flat = d_k.reshape(-1, d_k.shape[-1])
    d_v_flat = d_v.reshape(-1, d_v.shape[-1])
    
    dw_q = x_flat.T @ d_q_flat
    dw_k = x_flat.T @ d_k_flat
    dw_v = x_flat.T @ d_v_flat
    
    return {
        'dx': dx,
        'dw_q': dw_q,
        'dw_k': dw_k,
        'dw_v': dw_v
    }

# Step 116 - choose_attention_head_config
def choose_attention_head_config(d_model, n_heads):
    if d_model % n_heads != 0:
        raise ValueError()
    return {
        'n_heads': n_heads,
        'd_head': d_model // n_heads,
        'd_model': d_model
    }

# Step 117 - create_multihead_qkv_projections
def create_multihead_qkv_projections(d_model, scale=0.02):
    return {
        'Wq': scale_w_small(make_2d_random(d_model, d_model, 0), scale),
        'Wk': scale_w_small(make_2d_random(d_model, d_model, 1), scale),
        'Wv': scale_w_small(make_2d_random(d_model, d_model, 2), scale)
    }

# Step 118 - create_multihead_output_projection
def create_multihead_output_projection(d_model, scale=0.02):
    return scale_w_small(make_2d_random(d_model, d_model, 0), scale)

# Step 119 - reshape_to_heads
def reshape_to_heads(x, n_heads, d_head):
    B, T, d_model = x.shape
    return x.reshape(B, T, n_heads, d_head)

# Step 120 - transpose_heads_to_front
def transpose_heads_to_front(x):
    return x.transpose(0, 2, 1, 3)

# Step 121 - get_multihead_n_heads
def get_multihead_n_heads(config):
    return config['n_heads']

# Step 122 - get_multihead_sequence_length
def get_multihead_sequence_length(x):
    return get_array_shape(x)[1]

# Step 123 - compute_d_head
def compute_d_head(d_model, n_heads):
    if d_model % n_heads != 0:
        raise ValueError()
    return d_model // n_heads

# Step 124 - multihead_masked_softmax_scores
def multihead_masked_softmax_scores(scores, mask):
    B, n_heads, T, _ = scores.shape
    masked_scores = apply_causal_mask(scores, mask)
    flat_scores = masked_scores.reshape(B * n_heads * T, T)
    flat_probs = stable_softmax_2d_rowwise(flat_scores)
    return flat_probs.reshape(B, n_heads, T, T)

# Step 125 - multihead_weighted_sum
def multihead_weighted_sum(weights, v_heads):
    return np.matmul(weights, v_heads)

# Step 126 - transpose_heads_to_back
def transpose_heads_to_back(x):
    return x.transpose(0, 2, 1, 3)

# Step 127 - get_multihead_output_sequence_length
def get_multihead_output_sequence_length(x_heads_back):
    return x_heads_back.shape[1]

# Step 128 - merge_heads_to_d_model
def merge_heads_to_d_model(x_heads_back):
    B, T, n_heads, d_head = x_heads_back.shape
    d_model = n_heads * d_head
    return x_heads_back.reshape(B,T, d_model)

# Step 129 - multihead_output_projection_forward
def multihead_output_projection_forward(merged, w_out, b_out):
    linear_res = linear_forward(merged, w_out)
    bias_res = bias_add_forward(linear_res['y'], b_out)
    
    return {
        'out': bias_res['y'],
        'cache': {
            'merged': merged,
            'w_out': w_out
        }
    }

# Step 130 - multihead_reshape_transpose_backward
def multihead_reshape_transpose_backward(d_merged, shape_info):
    n_heads = shape_info['n_heads']
    d_head = shape_info['d_head']
    
    d_heads_back = reshape_to_heads(d_merged, n_heads, d_head)
    d_heads_front = transpose_heads_to_front(d_heads_back)
    
    return d_heads_front

# Step 131 - ffn_linear_one_forward
def ffn_linear_one_forward(x, w1, b1):
    linear_res = linear_forward(x, w1)
    bias_res = bias_add_forward(linear_res['y'], b1)
    
    return {
        'h1': bias_res['y'],
        'cache': {
            'x': x,
            'w1': w1
        }
    }

# Step 132 - ffn_activation_forward
def ffn_activation_forward(h1):
    relu_res = relu_forward(h1)
    
    return relu_res['y'], {'h1': h1}

# Step 133 - ffn_linear_two_forward
def ffn_linear_two_forward(a1, w2, b2):
    out_linear = linear_forward(a1, w2)
    h2_pre = next(v for k, v in out_linear.items() if k != 'cache')
    
    out_bias = bias_add_forward(h2_pre, b2)
    h2 = next(v for k, v in out_bias.items() if k != 'cache')
    
    return {
        'h2': h2,
        'cache': {
            'a1': a1,
            'w2': w2
        }
    }

# Step 134 - ffn_backward
def ffn_backward(d_out, cache):
    x = cache['x']
    w1 = cache['w1']
    h1 = cache['h1']
    a1 = cache['a1']
    w2 = cache['w2']
    
    B, T, _ = x.shape
    
    d_out_flat = d_out.reshape(B * T, -1)
    x_flat = x.reshape(B * T, -1)
    h1_flat = h1.reshape(B * T, -1)
    a1_flat = a1.reshape(B * T, -1)
    
    cache_2 = {'x': a1_flat, 'w': w2}
    cache_1 = {'x': x_flat, 'w': w1}
    cache_relu = {'x': h1_flat}
    
    def try_call(func, dout, array_arg, dict_arg):
        try:
            return func(dout, dict_arg)
        except Exception:
            return func(dout, array_arg)
            
    da1_flat = try_call(linear_backward_dx, d_out_flat, w2, cache_2)
    dw2 = try_call(linear_backward_dw, d_out_flat, a1_flat, cache_2)
    try:
        db2 = bias_add_backward_db(d_out_flat)
    except Exception:
        db2 = bias_add_backward_db(d_out_flat, cache_2)
        
    dh1_flat = try_call(relu_backward, da1_flat, h1_flat, cache_relu)
    dx_flat = try_call(linear_backward_dx, dh1_flat, w1, cache_1)
    dw1 = try_call(linear_backward_dw, dh1_flat, x_flat, cache_1)
    try:
        db1 = bias_add_backward_db(dh1_flat)
    except Exception:
        db1 = bias_add_backward_db(dh1_flat, cache_1)
        
    dx = dx_flat.reshape(x.shape)
    
    return {
        'dx': dx,
        'dw1': dw1,
        'db1': db1,
        'dw2': dw2,
        'db2': db2
    }

# Step 135 - residual_forward
def residual_forward(x, s):
    return x + s

# Step 136 - residual_backward
def residual_backward(d_y):
    return d_y.copy(), d_y.copy()

# Step 137 - pre_layernorm_sublayer_forward
def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params):
    eps = ln_params.get('eps', 1e-5)
    ln_res = layernorm_forward_affine(x, ln_params['gamma'], ln_params['beta'], eps)
    
    if isinstance(ln_res, dict):
        norm_x = ln_res.get('y', ln_res.get('out'))
        ln_cache = ln_res['cache']
    else:
        norm_x, ln_cache = ln_res
        
    sublayer_res = sublayer_fn(norm_x, sublayer_params)
    s = sublayer_res['y']
    sublayer_cache = sublayer_res['cache']
    
    y = residual_forward(x, s)
    
    return {
        'y': y,
        'cache': {
            'x': x,
            'ln_cache': ln_cache,
            'sublayer_cache': sublayer_cache
        }
    }

# Step 138 - transformer_block_forward
import numpy as np

def transformer_block_forward(x, block_params):
    
    def attn_wrapper(norm_x, params):
        Wq = params.get('Wq', params.get('w_q'))
        Wk = params.get('Wk', params.get('w_k'))
        Wv = params.get('Wv', params.get('w_v'))
        Wo = params.get('Wo', params.get('w_o', params.get('W_o', params.get('w_out'))))
        bo = params.get('bo', params.get('b_o', params.get('b_out')))
        n_heads = params['n_heads']
        
        B, T, d_model = norm_x.shape
        d_head = d_model // n_heads
        
        q = (norm_x @ Wq).reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
        k = (norm_x @ Wk).reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
        v = (norm_x @ Wv).reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
        
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(d_head)
        
        mask = np.tril(np.ones((T, T), dtype=bool))
        scores = np.where(mask, scores, -np.inf)
        
        row_max = np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores - row_max) / np.sum(np.exp(scores - row_max), axis=-1, keepdims=True)
        
        out_heads = probs @ v
        merged = out_heads.transpose(0, 2, 1, 3).reshape(B, T, d_model)
        
        y = merged @ Wo if Wo is not None else merged
        
        if bo is not None:
            y += bo
            
        return {
            'y': y,
            'cache': {
                'x': norm_x,
                'q': q, 'k': k, 'v': v,
                'probs': probs, 
                'merged': merged,
                'Wo': Wo
            }
        }

    def ffn_wrapper(norm_x, params):
        w1 = params.get('w1', params.get('W1'))
        b1 = params.get('b1', params.get('B1'))
        w2 = params.get('w2', params.get('W2'))
        b2 = params.get('b2', params.get('B2'))
        
        h1 = norm_x @ w1 + b1
        a1 = np.maximum(0, h1)
        y = a1 @ w2 + b2
        
        return {
            'y': y,
            'cache': {
                'x': norm_x, 
                'w1': w1, 'b1': b1, 
                'h1': h1, 'a1': a1, 
                'w2': w2, 'b2': b2
            }
        }

    attn_out = pre_layernorm_sublayer_forward(
        x, block_params['ln1'], attn_wrapper, block_params['attn']
    )
    
    ffn_out = pre_layernorm_sublayer_forward(
        attn_out['y'], block_params['ln2'], ffn_wrapper, block_params['ffn']
    )
    
    return {
        'y': ffn_out['y'],
        'cache': {
            'attn_branch': attn_out['cache'],
            'ffn_branch': ffn_out['cache']
        }
    }

# Step 139 - transformer_block_backward
import numpy as np

def transformer_block_backward(d_y, cache, block_params):
    x = cache['attn_branch']['x']
    full_cache = _complete_block_cache(x, block_params)
    
    d_ffn_out = d_y
    d_ln2_out, ffn_grads = _ffn_sublayer_backward(
        d_ffn_out, 
        full_cache['ffn_branch']['sublayer_cache'], 
        block_params['ffn']
    )
    
    d_ln2_in, d_ln2_gamma, d_ln2_beta = layernorm_backward_affine(
        d_ln2_out, 
        full_cache['ffn_branch']['ln_cache']
    )
    
    d_h1 = d_y + d_ln2_in
    
    d_attn_out = d_h1
    d_ln1_out, attn_grads = _attn_sublayer_backward(
        d_attn_out, 
        full_cache['attn_branch']['sublayer_cache'], 
        block_params['attn']
    )
    
    d_ln1_in, d_ln1_gamma, d_ln1_beta = layernorm_backward_affine(
        d_ln1_out, 
        full_cache['attn_branch']['ln_cache']
    )
    
    d_x = d_h1 + d_ln1_in
    
    grads = {
        'ln1': {'gamma': d_ln1_gamma, 'beta': d_ln1_beta},
        'ln2': {'gamma': d_ln2_gamma, 'beta': d_ln2_beta},
        'attn': attn_grads,
        'ffn': ffn_grads
    }
    
    return d_x, grads

# Step 140 - stack_transformer_blocks
def stack_transformer_blocks(n_layers, d_model, n_heads, d_ff):
    blocks = []
    for _ in range(n_layers):
        qkv = create_multihead_qkv_projections(d_model)
        Wo = create_multihead_output_projection(d_model)
        
        blocks.append({
            'ln1': {
                'gamma': np.ones(d_model),
                'beta': np.zeros(d_model)
            },
            'attn': {
                'Wq': qkv['Wq'],
                'Wk': qkv['Wk'],
                'Wv': qkv['Wv'],
                'Wo': Wo,
                'bo': np.zeros(d_model)
            },
            'ln2': {
                'gamma': np.ones(d_model),
                'beta': np.zeros(d_model)
            },
            'ffn': {
                'W1': scale_w_small(make_2d_random(d_model, d_ff, 0), 0.02),
                'b1': np.zeros(d_ff),
                'W2': scale_w_small(make_2d_random(d_ff, d_model, 1), 0.02),
                'b2': np.zeros(d_model)
            }
        })
        
    return blocks

# Step 141 - forward_through_all_blocks
def forward_through_all_blocks(x, blocks):
    h = x
    caches = []
    for block_params in blocks:
        out = transformer_block_forward(h, block_params)
        h = out['y']
        caches.append(out['cache'])
    return h, caches

# Step 142 - backward_through_all_blocks
def backward_through_all_blocks(d_y, caches, blocks):
    n_layers = len(blocks)
    grads = [None] * n_layers
    d_h = d_y
    for i in range(n_layers - 1, -1, -1):
        d_h, grads_block = transformer_block_backward(d_h, caches[i], blocks[i])
        grads[i] = grads_block
    return d_h, grads

# Step 143 - final_layernorm_forward
def final_layernorm_forward(x, gamma, beta):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + 1e-5)
    y = x_hat * gamma + beta
    
    cache = {
        'x': x,
        'mean': mean,
        'var': var,
        'x_hat': x_hat,
        'gamma': gamma
    }
    
    return y, cache

# Step 144 - lm_head_linear_forward
def lm_head_linear_forward(x, w_lm, b_lm):
    linear_out = linear_forward(x, w_lm)
    bias_out = bias_add_forward(linear_out['y'], b_lm)
    
    return {
        'logits': bias_out['y'],
        'cache': {
            'x': x,
            'w_lm': w_lm
        }
    }

# Step 145 - full_model_forward
def full_model_forward(ids, model_params):
    tok_out, tok_cache = token_embedding_forward(ids, model_params['tok_emb'])
    pos_emb_sliced = slice_positional_embedding(model_params['pos_emb'], ids.shape[1])
    emb_sum = add_token_and_positional_embeddings(tok_out, pos_emb_sliced)
    
    blocks_out, blocks_caches = forward_through_all_blocks(emb_sum, model_params['blocks'])
    
    ln_f_out, ln_f_cache = final_layernorm_forward(
        blocks_out,
        model_params['ln_f']['gamma'],
        model_params['ln_f']['beta']
    )
    
    lm_head_out = lm_head_linear_forward(
        ln_f_out,
        model_params['lm_head']['w_lm'],
        model_params['lm_head']['b_lm']
    )
    
    caches = {
        'emb': tok_cache,
        'blocks': blocks_caches,
        'ln_f': ln_f_cache,
        'lm_head': lm_head_out['cache']
    }
    
    return lm_head_out['logits'], caches

# Step 146 - full_model_backward
def full_model_backward(d_logits, caches, model_params):
    x_lm = caches['lm_head']['x']
    w_lm = caches['lm_head']['w_lm']
    
    B, T, V = d_logits.shape
    D = x_lm.shape[-1]
    
    dx_lm = d_logits @ w_lm.T
    dw_lm = x_lm.reshape(-1, D).T @ d_logits.reshape(-1, V)
    db_lm = np.sum(d_logits, axis=(0, 1))
    
    dy = dx_lm
    x_hat = caches['ln_f']['x_hat']
    gamma = caches['ln_f']['gamma']
    var = caches['ln_f']['var']
    
    dgamma = np.sum(dy * x_hat, axis=(0, 1))
    dbeta = np.sum(dy, axis=(0, 1))
    
    dx_hat = dy * gamma
    std = np.sqrt(var + 1e-5)
    mean_dx_hat = np.mean(dx_hat, axis=-1, keepdims=True)
    mean_dx_hat_x_hat = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
    
    dx_ln = (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat) / std
    
    d_emb_sum, blocks_grads = backward_through_all_blocks(dx_ln, caches['blocks'], model_params['blocks'])
    
    d_tok_emb = np.zeros_like(model_params['tok_emb'])
    token_ids = caches['emb']['tok_cache']['token_ids'] if 'tok_cache' in caches['emb'] else caches['emb']['token_ids']
    np.add.at(d_tok_emb, token_ids, d_emb_sum)
    
    d_pos_emb = np.zeros_like(model_params['pos_emb'])
    seq_len = caches['emb']['seq_len'] if 'seq_len' in caches['emb'] else d_emb_sum.shape[1]
    d_pos_emb[:seq_len] = np.sum(d_emb_sum, axis=0)
    
    return {
        'tok_emb': d_tok_emb,
        'pos_emb': d_pos_emb,
        'blocks': blocks_grads,
        'ln_f': {
            'gamma': dgamma,
            'beta': dbeta
        },
        'lm_head': {
            'w_lm': dw_lm,
            'b_lm': db_lm
        }
    }

# Step 147 - initialize_adam_moments
import numpy as np

def initialize_adam_moments(model_params):
    def build_zeros(params):
        if isinstance(params, dict):
            return {k: build_zeros(v) for k, v in params.items()}
        elif isinstance(params, list):
            return [build_zeros(v) for v in params]
        elif isinstance(params, np.ndarray):
            return np.zeros_like(params)
        return params

    m = build_zeros(model_params)
    v = build_zeros(model_params)
    
    return m, v

# Step 148 - initialize_adam_step_counter
def initialize_adam_step_counter():
    return 0

# Step 149 - adam_increment_step
def adam_increment_step(t):
    return t + 1

# Step 150 - adam_update_first_moment
def adam_update_first_moment(m, grad, beta1):
    return beta1 * m + (1.0 - beta1) * grad

# Step 151 - adam_update_second_moment
def adam_update_second_moment(v_prev, grad, beta2):
    return beta2 * v_prev + (1.0 - beta2) * (grad * grad)

# Step 152 - adam_bias_correction
def adam_bias_correction(m, v, beta1, beta2, t):
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    return m_hat, v_hat

# Step 153 - adam_parameter_update
def adam_parameter_update(param, m_hat, v_hat, lr, eps):
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)

# Step 154 - wire_full_training_loop
def wire_full_training_loop(params, train_ids, val_ids, block_size, batch_size, n_steps, lr, betas, eps):
    rng = np.random.default_rng()
    t = initialize_adam_step_counter()
    m, v = initialize_adam_moments(params)
    beta1, beta2 = betas
    
    history = []
    
    def update_tree(p_node, g_node, m_node, v_node, current_t):
        if isinstance(p_node, dict):
            new_p, new_m, new_v = {}, {}, {}
            for k in p_node:
                new_p[k], new_m[k], new_v[k] = update_tree(
                    p_node[k], g_node[k], m_node[k], v_node[k], current_t
                )
            return new_p, new_m, new_v
        elif isinstance(p_node, list):
            new_p, new_m, new_v = [], [], []
            for i in range(len(p_node)):
                res_p, res_m, res_v = update_tree(
                    p_node[i], g_node[i], m_node[i], v_node[i], current_t
                )
                new_p.append(res_p)
                new_m.append(res_m)
                new_v.append(res_v)
            return new_p, new_m, new_v
        elif isinstance(p_node, np.ndarray):
            new_m_leaf = adam_update_first_moment(m_node, g_node, beta1)
            new_v_leaf = adam_update_second_moment(v_node, g_node, beta2)
            m_hat, v_hat = adam_bias_correction(new_m_leaf, new_v_leaf, beta1, beta2, current_t)
            new_p_leaf = adam_parameter_update(p_node, m_hat, v_hat, lr, eps)
            return new_p_leaf, new_m_leaf, new_v_leaf
        return p_node, m_node, v_node

    for step in range(n_steps):
        X, Y = get_batch(train_ids, block_size, batch_size, rng)
        
        logits, caches = full_model_forward(X, params)
        B, T, V = logits.shape
        N = B * T
        
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        Y_onehot = np.eye(V)[Y]
        
        loss = -np.sum(Y_onehot * np.log(probs + 1e-12)) / N
        
        d_logits = (probs - Y_onehot) / N
        
        grads = full_model_backward(d_logits, caches, params)
        
        t = adam_increment_step(t)
        params, m, v = update_tree(params, grads, m, v, t)
        
        history.append({'step': step, 'train_loss': loss})
        
    return params, history

# Step 155 - logging_and_validation_loss (not yet solved)
# TODO: implement

# Step 156 - encode_prompt (not yet solved)
# TODO: implement

# Step 157 - crop_context_to_block_size (not yet solved)
# TODO: implement

# Step 158 - forward_to_get_logits (not yet solved)
# TODO: implement

# Step 159 - take_last_position_logits (not yet solved)
# TODO: implement

# Step 160 - apply_temperature (not yet solved)
# TODO: implement

# Step 161 - top_k_filter (not yet solved)
# TODO: implement

# Step 162 - softmax_to_probs (not yet solved)
# TODO: implement

# Step 163 - sample_one_token (not yet solved)
# TODO: implement

# Step 164 - append_token_to_sequence (not yet solved)
# TODO: implement

# Step 165 - generation_loop_for_n_steps (not yet solved)
# TODO: implement

# Step 166 - decode_final_sequence (not yet solved)
# TODO: implement

