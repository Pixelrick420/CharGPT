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

# Step 123 - compute_d_head (not yet solved)
# TODO: implement

# Step 124 - multihead_masked_softmax_scores (not yet solved)
# TODO: implement

# Step 125 - multihead_weighted_sum (not yet solved)
# TODO: implement

# Step 126 - transpose_heads_to_back (not yet solved)
# TODO: implement

# Step 127 - get_multihead_output_sequence_length (not yet solved)
# TODO: implement

# Step 128 - merge_heads_to_d_model (not yet solved)
# TODO: implement

# Step 129 - multihead_output_projection_forward (not yet solved)
# TODO: implement

# Step 130 - multihead_reshape_transpose_backward (not yet solved)
# TODO: implement

# Step 131 - ffn_linear_one_forward (not yet solved)
# TODO: implement

# Step 132 - ffn_activation_forward (not yet solved)
# TODO: implement

# Step 133 - ffn_linear_two_forward (not yet solved)
# TODO: implement

# Step 134 - ffn_backward (not yet solved)
# TODO: implement

# Step 135 - residual_forward (not yet solved)
# TODO: implement

# Step 136 - residual_backward (not yet solved)
# TODO: implement

# Step 137 - pre_layernorm_sublayer_forward (not yet solved)
# TODO: implement

# Step 138 - transformer_block_forward (not yet solved)
# TODO: implement

# Step 139 - transformer_block_backward (not yet solved)
# TODO: implement

# Step 140 - stack_transformer_blocks (not yet solved)
# TODO: implement

# Step 141 - forward_through_all_blocks (not yet solved)
# TODO: implement

# Step 142 - backward_through_all_blocks (not yet solved)
# TODO: implement

# Step 143 - final_layernorm_forward (not yet solved)
# TODO: implement

# Step 144 - lm_head_linear_forward (not yet solved)
# TODO: implement

# Step 145 - full_model_forward (not yet solved)
# TODO: implement

# Step 146 - full_model_backward (not yet solved)
# TODO: implement

# Step 147 - initialize_adam_moments (not yet solved)
# TODO: implement

# Step 148 - initialize_adam_step_counter (not yet solved)
# TODO: implement

# Step 149 - adam_increment_step (not yet solved)
# TODO: implement

# Step 150 - adam_update_first_moment (not yet solved)
# TODO: implement

# Step 151 - adam_update_second_moment (not yet solved)
# TODO: implement

# Step 152 - adam_bias_correction (not yet solved)
# TODO: implement

# Step 153 - adam_parameter_update (not yet solved)
# TODO: implement

# Step 154 - wire_full_training_loop (not yet solved)
# TODO: implement

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

