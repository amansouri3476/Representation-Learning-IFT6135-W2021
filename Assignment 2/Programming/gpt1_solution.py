import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.embeddings import GPT1Embedding


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def forward(self, inputs):
        """Layer Normalization.

        This module applies Layer Normalization, with rescaling and shift,
        only on the last dimension. See Lecture 07 (I), slide 23.

        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The input tensor. This tensor can have an arbitrary number N of
            dimensions, as long as `inputs.shape[N-1] == hidden_size`. The
            leading N - 1 dimensions `dims` can be arbitrary.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The output tensor, having the same shape as `inputs`.
        """

        # ==========================
        # TODO: Write your code here
        
#         mu = inputs.mean(tuple(range(len(inputs.size())-1)),keepdim=True)
#         sigma_2 = inputs.var(tuple(range(len(inputs.size())-1)),keepdim=True, unbiased=True)
        
        mu = inputs.mean(-1,keepdim=True)
        sigma_2 = inputs.var(-1,keepdim=True, unbiased=False)

        layernorm = (inputs-mu)*self.weight/(torch.sqrt(sigma_2+self.eps))+self.bias

        return layernorm
    
    
        # ==========================
        


    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size, num_heads, sequence_length):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length

        # ==========================
        # TODO: Write your code here


        self.W_Q = nn.Parameter(torch.Tensor(num_heads*head_size,num_heads*head_size))
        self.W_V = nn.Parameter(torch.Tensor(num_heads*head_size,num_heads*head_size))
        self.W_K = nn.Parameter(torch.Tensor(num_heads*head_size,num_heads*head_size))
        
        self.b_Q = nn.Parameter(torch.Tensor(num_heads*head_size))
        self.b_V = nn.Parameter(torch.Tensor(num_heads*head_size))
        self.b_K = nn.Parameter(torch.Tensor(num_heads*head_size))
        
        self.W_O = nn.Parameter(torch.Tensor(num_heads*head_size,num_heads*head_size))
        
        self.b_O = nn.Parameter(torch.Tensor(num_heads*head_size))
        
        self.weights = [self.W_Q,self.W_K,self.W_V,self.W_O]
        self.biases = [self.b_Q,self.b_K,self.b_V,self.b_O]

        self.reset_parameters()

        # ==========================
    
    
    def reset_parameters(self):
        for weight in self.weights:
#             nn.init.ones_(weight)
            nn.init.xavier_uniform_(weight)
        for bias in self.biases:
            nn.init.zeros_(bias)
#             torch.nn.init.xavier_uniform(bias)
        
        
    def get_attention_weights(self, queries, keys):
        """Compute the attention weights.

        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for
        simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as

            weights = softmax(Q * K^{T} / sqrt(head_size))

        Here "*" is the matrix multiplication. Your attention weights must
        take into account the fact that we have a causal language model, i.e.
        there should be no influence from the future, attention is only
        computed on the past. See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. For example, `queries[1, 3, 5]` is the query of
            the 4th head (index 3) for the 6th token (index 5) in the 2nd
            sequence (index 1) in the batch (it is a vector of size `head_size`).

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. For example, `keys[0, 2, 4]` is the key of the
            3rd head (index 2) for the 5th token (index 4) in the 1st sequence
            (index 0) in the batch (it is a vector of size `head_size`).

        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch. For example, `attention_weights[1, 3, 5, 7]`
            is the attention weights from the 8th token (index 7) on the 6th
            token (index 5) of the 4th head (index 3) in the 2nd sequence
            (index 1) in the batch. Note that because we have a causal language
            model here, `attention_weights[1, 3, 5, 7] == 0`, since the 8th token
            should not influence on the 6th token (7 > 5).
        """

        # ==========================
        # TODO: Write your code here
        
        keys_transposed = keys.permute(0,1,3,2)
        
        mask_1 = torch.ones(self.sequence_length,self.sequence_length, device=keys.device)
        mask_1 = mask_1.tril()
        
        mask_2 = torch.ones(self.sequence_length,self.sequence_length, device=keys.device)*-1e4
        mask_2 = mask_2.triu(diagonal=1)
        
        dot_product = torch.matmul(queries,keys_transposed)
        
        masked_weights = (dot_product*mask_1/np.sqrt(self.head_size))+mask_2
        
        return F.softmax(masked_weights,dim=-1)
    
        # ==========================
        

        

    def apply_attention(self, queries, keys, values):
        """Apply the attention.

        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by

            weights = softmax(Q * K^{T} / sqrt(head_size))
            attended_values = weights * V
            outputs = concat(attended_values)

        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. For example, `queries[1, 3, 5]` is the query of
            the 4th head (index 3) for the 6th token (index 5) in the 2nd
            sequence (index 1) in the batch (it is a vector of size `head_size`).

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. For example, `keys[0, 2, 4]` is the key of the
            3rd head (index 2) for the 5th token (index 4) in the 1st sequence
            (index 0) in the batch (it is a vector of size `head_size`).

        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads. For example, `values[1, 3, 5]` is the key of the
            4th head (index 3) for the 6th token (index 5) in the 2nd sequence
            (index 1) in the batch (it is a vector of size `head_size`).

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence. For
            example, `outputs[0, 2]` contains the output of the attention
            (concatenated for all heads) for the 3rd token (index 2) of the 1st
            sequence in the batch (index 0).
        """

        # ==========================
        # TODO: Write your code here
        
                
        attention_weights = self.get_attention_weights(queries, keys)
        
        hs = torch.matmul(attention_weights,values)
        
        hs_transposed = hs.permute(0,2,1,3)
        
        shapes = hs_transposed.size()
        
        attention_weights = self.get_attention_weights(queries, keys)
        
        hs = torch.matmul(attention_weights,values)
        
        hs_transposed = hs.permute(0,2,1,3)
        
        shapes = hs_transposed.size()
        
        return hs_transposed.contiguous().view(shapes[0],shapes[1],shapes[2]*shapes[3])
    
        # ==========================

        
    def split_heads(self, tensor):
        """Split the head vectors.

        This function splits the head vectors that have been concatenated (e.g.
        through the `merge_heads` function) into a separate dimension. This
        function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not
        apply any further transformation to the tensor. The function `split_heads`
        is the inverse of the function `merge_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        # ==========================
        # TODO: Write your code here
        
                
        batch_size = tensor.shape[0]
        seq_len = tensor.shape[1]
        last_dim = tensor.shape[2]
        
        return tensor.view(batch_size,seq_len,self.num_heads,int(last_dim/self.num_heads)).permute(0,2,1,3)

        # ==========================
        

    def merge_heads(self, tensor):
        """Merge the head vectors.

        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        # ==========================
        # TODO: Write your code here
        
                
        batch_size = tensor.shape[0]
        n_heads = tensor.shape[1]
        seq_len = tensor.shape[2]
        dim = tensor.shape[3]
        
        temp_tensor = tensor.permute((0,2,1,3))
        
        return temp_tensor.contiguous().view(batch_size,seq_len,n_heads*dim)

        # ==========================

    def forward(self, hidden_states):
        """Multi-headed attention.

        This applies the multi-headed attention on the input tensors `hidden_states`.
        For a single sequence (for simplicity), if X are the hidden states from
        the previous layer (a matrix of size `(sequence_length, num_heads * head_size)`
        containing the concatenated head vectors), then the output of multi-headed
        attention is given by

            Q = X * W_{Q} + b_{Q}        # Queries
            K = X * W_{K} + b_{K}        # Keys
            V = X * W_{V} + b_{V}        # Values

            Y = attention(Q, K, V)       # Attended values (concatenated for all heads)
            outputs = Y * W_{Y} + b_{Y}  # Linear projection

        Here "*" is the matrix multiplication.

        Parameters
        ----------
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Input tensor containing the concatenated head vectors for all the
            sequences in the batch, and all positions in each sequence. This
            is, for example, the tensor returned by the previous layer.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        """

        # ==========================
        # TODO: Write your code here
        
        X = hidden_states
        
        Q = torch.matmul(X,self.W_Q) + self.b_Q
        K = torch.matmul(X,self.W_K) + self.b_K
        V = torch.matmul(X,self.W_V) + self.b_V
        
        Y = self.apply_attention(self.split_heads(Q), self.split_heads(K), self.split_heads(V))
        
        return torch.matmul(Y,self.W_O)+self.b_O
        # ==========================
        


class Block(nn.Module):
    def __init__(self, head_size, mlp_hidden_size, num_heads, sequence_length):
        """A decoder layer.

        This module combines a Multi-headed Attention module and an MLP to
        create a layer of the transformer, with normalization and skip-connections.
        See Lecture 06, slide 33.
        """
        super(Block, self).__init__()
        self.head_size = head_size
        self.mlp_hidden_size = mlp_hidden_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.hidden_size = num_heads * head_size

        self.attention = MultiHeadedAttention(head_size, num_heads, sequence_length)
        self.norm1 = LayerNorm(self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, self.hidden_size),
        )
        self.norm2 = LayerNorm(self.hidden_size)

    def forward(self, hidden_states):
        attention_outputs = self.attention(hidden_states)
        attention_outputs = self.norm1(attention_outputs + hidden_states)
        outputs = self.mlp(attention_outputs)
        outputs = self.norm2(outputs + attention_outputs)
        return outputs


class MiniGPT1(nn.Module):
    def __init__(
        self,
        vocabulary_size=40479,
        embedding_size=768,
        sequence_length=256,
        num_heads=12,
        num_layers=4,
        learn_embeddings=False,
        _tokens_embedding_weight=None,
        _positional_embedding_weight=None,
    ):

        super(MiniGPT1, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.learn_embeddings = learn_embeddings
        self.head_size = embedding_size // num_heads

        self.embedding = GPT1Embedding(
            vocabulary_size,
            embedding_size,
            sequence_length,
            _tokens_embedding_weight=_tokens_embedding_weight,
            _positional_embedding_weight=_positional_embedding_weight,
        )
        self.layers = nn.ModuleList(
            [
                Block(self.head_size, 4 * embedding_size, num_heads, sequence_length)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(embedding_size, vocabulary_size, bias=False)

        # Tying classifier and embedding weights
        self.classifier.weight = self.embedding.tokens.weight

        # Freeze the embedding weights, depending on learn_embeddings
        self.embedding.requires_grad_(learn_embeddings)

    def get_embeddings(self, inputs):
        """Get the embeddings for some input sequence.

        This function computes the embedding vectors based on the input
        sequence (and the positions of the tokens). See also the module
        `GPT1Embedding` for details about the implementation of
        `self.embedding`.

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        Returns
        -------
        embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, embedding_size)`)
            The tensor containing the embeddings. For example, `embeddings[0, 2]`
            is the embedding vector for the token in 3rd position (index 2)
            of the 1st sequence in the batch (index 0).
        """

        # ==========================
        # TODO: Write your code here
        
        batch_size = seq_len = inputs.shape[0]
        seq_len = inputs.shape[1]
        
        indices = np.arange(seq_len)
        indices_repeated = torch.tensor(np.tile(indices,(batch_size,1)), dtype=torch.long, device=inputs.device)

        return self.embedding.forward(inputs,indices_repeated)
    
        # ==========================
      

    def forward(self, inputs):
        """Mini GPT-1.

        This is a small version of OpenAI's GPT-1 transformer for (causal)
        language modeling. This module returns for each position in the
        sequence the log-probabilities of the next token.

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        Returns
        -------
        log_probas (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocabulary_size)`)
            A tensor containing the log-probabilities of the next token for
            all positions in each sequence of the batch. For example, `log_probas[0, 3, 6]`
            corresponds to log p(x_{5} = token_{7} | x_{0:4}) (x_{5} for the word
            after x_{4} at index 3, and token_{7} for index 6) for the 1st sequence
            of the batch (index 0).
        """

        # ==========================
        # TODO: Write your code here
        
        embedded_inputs = self.get_embeddings(inputs)
        
        next_block_input = embedded_inputs
        for layer in self.layers:
            next_block_input = layer(next_block_input)
            

        mlp_output = self.classifier(next_block_input)
        
        # Taking the log-softmax to obtain log-probabilities.
        log_probas = F.log_softmax(mlp_output,2)
        
        return log_probas
        # ==========================
        

    def loss(self, log_probas, targets, mask):
        """Loss function.

        This function computes the loss (negative log-likelihood).

        Parameters
        ----------
        log_probas (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocabulary_size)`)
            A tensor containing the log-probabilities of the next token for
            all positions in each sequence of the batch.

        targets (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            A tensor containing the target next tokens for all positions in
            each sequence of the batch.

        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`)
            A tensor containing values in {0, 1} only, where the value is 0
            for positions corresponding to padding in the sequence, and 1
            otherwise.

        Returns
        -------
        loss (`torch.FloatTensor` scalar)
            The scalar loss, corresponding to the (mean) negative log-likelihood.
        """

        # ==========================
        # TODO: Write your code here
        
        # Reshaping the log_probas to (batch*seq_len,|Vocab|) so that it becomes compatible with nll_loss
        shapes = log_probas.size()
        log_probas_reshaped = log_probas.view(shapes[0]*shapes[1],shapes[2])
        
        # Computing the nll_loss WITHOUT reduction
        all_loss = F.nll_loss(log_probas_reshaped, targets.reshape(-1), reduction='none')
        
        all_loss_reshaped = all_loss.view(shapes[0],shapes[1])
        
        losses_masked = all_loss_reshaped*mask/mask.sum(1, keepdim=True)
        
        return losses_masked.sum()/shapes[0] # Only divide by batchsize because avg over masks=1 has been taken care of.
    
        # ==========================
        

    @classmethod
    def load_embeddings_from(
        cls, filename, num_heads=12, num_layers=4, learn_embeddings=False
    ):
        # Load the embeddings from filename
        with open(filename, "rb") as f:
            embeddings = np.load(f)
            tokens_weight = torch.from_numpy(embeddings["tokens"])
            positional_weight = torch.from_numpy(embeddings["position"])

        vocabulary_size, embedding_size = tokens_weight.shape
        sequence_length = positional_weight.size(0)
        return cls(
            vocabulary_size,
            embedding_size,
            sequence_length,
            num_heads,
            num_layers,
            learn_embeddings,
            _tokens_embedding_weight=tokens_weight,
            _positional_embedding_weight=positional_weight,
        )
