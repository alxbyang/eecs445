import numpy as np


class TinyTransformer:
    def __init__(self, embed_dim, d_k, d_v):
        """
        Args:
        embed_dim: Dimensionality of input embeddings
        d_k: Dimensionality of Query/Key
        d_v: Dimensionality of Value
        """
        self.embed_dim = embed_dim
        self.d_k = d_k
        self.d_v = d_v

        # ================== Do not modify (below) ==================
        self.W_Q = np.array([[1.0, 0.9], [0.5, 0.8]])  # Shape [embed_dim, d_k]

        self.W_K = np.array([[0.9, 0.9], [0.6, 0.7]])  # Shape [embed_dim, d_k]

        self.W_V = np.array([[0.8, 0.9], [0.7, 0.85]])  # Shape [embed_dim, d_v]

        self.W_O = np.array([[0.4, 0.3], [0.6, 0.5]])  # Shape [d_v, embed_dim] (output projection)
        # ================== Do not modify (above) ==================


    def compute_qkv(self, X):
        """
        Compute Q, K, and V matrices using linear transformations.
        Args:
        X: Input embeddings, shape [batch_size, seq_len, embed_dim]
        
        Returns:
        Q, K, V matrices
        """
        # TODO: Compute Q, K, V matrices using linear algebra operations
        return Q, K, V


    def attention_scores(self, Q, K):
        """
        Compute scaled attention scores.
        Args:
        Q: Query matrix, shape [batch_size, seq_len, d_k]
        K: Key matrix, shape [batch_size, seq_len, d_k]
        
        Returns:
        Scaled attention scores, shape [batch_size, seq_len, seq_len]
        """
        # The Score function is defined as the matrix product of Q and K, divided by the square root of d_k
        # NOTE: the batch_size is 1 in this toy example
        # TODO Compute the attention scores
        scores = None

        # Print the attention scores for the word 'won' (index 1 in this example)
        print("--------------q2.a--------------")
        print("attention scores for the query word 'won': ", scores[0, 1, :])
        print("--------------------------------\n")
        return scores


    def attention_weights(self, scores):
        """
        Apply softmax to the attention scores to get attention weights.
        Args:
        scores: Attention scores, shape [batch_size, seq_len, seq_len]
        
        Returns:
        Normalized attention weights, shape [batch_size, seq_len, seq_len]
        """
        # TODO: first subtract the maximum value in each row for numerical stability
        # then use the softmax function to compute the attention weights
        weights = None
        print("--------------q2.b--------------")
        print("attention weights for the query word 'won': ", weights[0, 1, :])
        print("--------------------------------\n")
        return weights


    def context_vector(self, weights, V):
        """
        Compute context vector as a weighted sum of the value vectors.
        Args:
        weights: Attention weights, shape [batch_size, seq_len, seq_len]
        V: Value matrix, shape [batch_size, seq_len, d_v]
        
        Returns:
        Context vector, shape [batch_size, seq_len, d_v]
        """
        # TODO: Compute the final context vector
        context = None
        print("--------------q2.b--------------")
        print("context vector for the word 'won': ", context[0, 1, :])
        print("--------------------------------\n")
        return context


    def single_head_attention(self, Q, K, V):
        """
        Perform single-head attention (simplified for Tiny Transformer).
        Args:
        Q: Query matrix, shape [batch_size, seq_len, d_k]
        K: Key matrix, shape [batch_size, seq_len, d_k]
        V: Value matrix, shape [batch_size, seq_len, d_v]
        
        Returns:
        Context vector after attention, shape [batch_size, seq_len, d_v]
        """
        scores = self.attention_scores(Q, K)
        weights = self.attention_weights(scores)
        context = self.context_vector(weights, V)
        return context


    def residual_connection_and_layer_norm(self, X, attention_output, epsilon=1e-6):
        """
        Apply residual connection and layer normalization.
        Args:
        X: Original input embeddings, shape [batch_size, seq_len, embed_dim]
        attention_output: Output from the attention mechanism, shape [batch_size, seq_len, embed_dim]
        
        Returns:
        Output after residual connection and layer normalization
        """
        residual_output = X + attention_output  # Residual connection
        mean = np.mean(residual_output, axis=-1, keepdims=True)
        variance = np.var(residual_output, axis=-1, keepdims=True)
        normalized_output = (residual_output - mean) / np.sqrt(variance + epsilon)  # Layer normalization
        return normalized_output


    def forward(self, X):
        """
        Forward pass for the Tiny Transformer.
        Args:
        X: Input embeddings, shape [batch_size, seq_len, embed_dim]
        
        Returns:
        Final output from the transformer, shape [batch_size, seq_len, embed_dim]
        """
        # Step 1: Compute Q, K, V
        Q, K, V = self.compute_qkv(X)

        print("--------Compare your result with Q, K, V given in the question---------")
        print("Matrix Q: \n", Q)
        print("Matrix K: \n", K)
        print("Matrix V: \n", V)
        print("--------------------------------\n")
        
        # Step 2: Perform single-head attention
        attention_output = self.single_head_attention(Q, K, V)
        
        # Step 3: Apply residual connection and layer normalization
        final_output = self.residual_connection_and_layer_norm(X, attention_output)
        
        return final_output


if __name__ == "__main__":
    # ================== Do not modify (below) ==================
    # Set up the parameters for the Tiny Transformer
    batch_size = 1
    seq_len = 4
    embed_dim = 2  # Smaller embedding dimension for simplicity
    d_k = 2  # Dimension of Query/Key
    d_v = 2  # Dimension of Value

    # Create random input embeddings (batch of sequences)
    X = np.array([[[1.0, 0.5], [0.9, 0.8], [0.2, 0.1], [0.8, 1.0]]])
    # ================== Do not modify (above) ==================

    # Initialize the Tiny Transformer
    transformer = TinyTransformer(embed_dim, d_k, d_v)

    # Forward pass
    output = transformer.forward(X)

    # Print the final output
    print("Final output from Tiny Transformer:\n", output)
