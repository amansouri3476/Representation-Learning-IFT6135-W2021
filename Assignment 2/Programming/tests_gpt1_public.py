import unittest
import torch

from gradescope_utils.autograder_utils.decorators import weight

from gpt1_solution import LayerNorm, MultiHeadedAttention, MiniGPT1


class LayerNormShapes(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 11

        self.layer_norm = LayerNorm(self.hidden_size)

    @weight(0)
    def test_forward_shape(self):
        shape = (3, 5, 7)
        inputs = torch.randn(*shape, self.hidden_size)

        outputs = self.layer_norm(inputs)

        expected_shape = (*shape, self.hidden_size)
        assert isinstance(outputs, torch.Tensor), (
            "The output of the module " "must be a torch.Tensor"
        )
        assert outputs.shape == expected_shape, (
            "The shape of the output of "
            "the module is invalid with `inputs` of shape {0}.\n  Got "
            "shape {1}\n  Expected shape: {2}\nRecall that the output should "
            "have shape (*dims, hidden_size)".format(
                tuple(inputs.shape), tuple(outputs.shape), expected_shape
            )
        )


class TestMultiHeadedAttentionShapes(unittest.TestCase):
    def setUp(self):
        self.head_size = 11
        self.num_heads = 13
        self.sequence_length = 17

        self.attention = MultiHeadedAttention(
            self.head_size, self.num_heads, self.sequence_length
        )

    @weight(0)
    def test_get_attention_weights_shape(self):
        batch_size = 7
        queries = torch.randn(
            batch_size, self.num_heads, self.sequence_length, self.head_size
        )
        keys = torch.randn(
            batch_size, self.num_heads, self.sequence_length, self.head_size
        )

        attention_weights = self.attention.get_attention_weights(queries, keys)

        expected_shape = (
            batch_size,
            self.num_heads,
            self.sequence_length,
            self.sequence_length,
        )
        assert isinstance(attention_weights, torch.Tensor), (
            "The output of " "`get_attention_weights` must be a torch.Tensor."
        )
        assert attention_weights.shape == expected_shape, (
            "The shape of your "
            "attention weights is invalid with `queries` of shape {0}, and "
            "`keys` of shape {1}.\n  Got shape: {2}\n  Expected shape: {3}\n"
            "Recall that the attention weights should have shape (batch_size, "
            "num_heads, sequence_length, sequence_length).".format(
                tuple(queries.shape),
                tuple(keys.shape),
                tuple(attention_weights.shape),
                expected_shape,
            )
        )

    @weight(0)
    def test_apply_attention_shape(self):
        batch_size = 7
        queries = torch.randn(
            batch_size, self.num_heads, self.sequence_length, self.head_size
        )
        keys = torch.randn(
            batch_size, self.num_heads, self.sequence_length, self.head_size
        )
        values = torch.randn(
            batch_size, self.num_heads, self.sequence_length, self.head_size
        )

        outputs = self.attention.apply_attention(queries, keys, values)

        expected_shape = (
            batch_size,
            self.sequence_length,
            self.num_heads * self.head_size,
        )
        assert isinstance(outputs, torch.Tensor), (
            "The output of " "`apply_attention` must be a torch.Tensor."
        )
        assert outputs.shape == expected_shape, (
            "The shape of the output of "
            "`apply_attention` is invalid with `queries` of shape {0}, `keys` "
            "of shape {1}, and `values` of shape {2}.\n  Got shape: {3}\n  "
            "Expected shape: {4}\nRecall that the attention weights should have "
            "shape (batch_size, sequence_length, num_heads * head_size).".format(
                tuple(queries.shape),
                tuple(keys.shape),
                tuple(values.shape),
                tuple(outputs.shape),
                expected_shape,
            )
        )

    @weight(0)
    def test_split_heads_shape(self):
        batch_size = 7
        dim = 23
        tensor = torch.randn(batch_size, self.sequence_length, self.num_heads * dim)

        output = self.attention.split_heads(tensor)

        expected_shape = (batch_size, self.num_heads, self.sequence_length, dim)
        assert isinstance(output, torch.Tensor), (
            "The output of `split_heads` " "must be a torch.Tensor."
        )
        assert output.shape == expected_shape, (
            "The shape of the output of "
            "`split_heads` is invalid with `tensor` of shape {0}.\n  Got shape "
            "{1}\n  Expected shape: {2}\nRecall that the output should have "
            "shape (batch_size, num_heads, sequence_length, dim)".format(
                tuple(tensor.shape), tuple(output.shape), expected_shape
            )
        )

    @weight(0)
    def test_merge_heads_shape(self):
        batch_size = 7
        dim = 23
        tensor = torch.randn(batch_size, self.num_heads, self.sequence_length, dim)

        output = self.attention.merge_heads(tensor)

        expected_shape = (batch_size, self.sequence_length, self.num_heads * dim)
        assert isinstance(output, torch.Tensor), (
            "The output of `merge_heads` " "must be a torch.Tensor."
        )
        assert output.shape == expected_shape, (
            "The shape of the output of "
            "`merge_heads` is invalid with `tensor` of shape {0}.\n  Got shape "
            "{1}\n  Expected shape: {2}\nRecall that the output should have "
            "shape (batch_size, sequence_length, num_heads * dim)".format(
                tuple(tensor.shape), tuple(output.shape), expected_shape
            )
        )

    @weight(0)
    def test_forward_shape(self):
        batch_size = 7
        hidden_states = torch.randn(
            batch_size, self.sequence_length, self.num_heads * self.head_size
        )

        outputs = self.attention(hidden_states)

        expected_shape = (
            batch_size,
            self.sequence_length,
            self.num_heads * self.head_size,
        )
        assert isinstance(outputs, torch.Tensor), (
            "The output of the module " "must be a torch.Tensor."
        )
        assert outputs.shape == expected_shape, (
            "The shape of the output of "
            "the module is invalid with `hidden_sizes` of shape {0}.\n  Got "
            "shape {1}\n  Expected shape: {2}\nRecall that the output should "
            "have shape (batch_size, sequence_length, num_heads * head_size)".format(
                tuple(hidden_states.shape), tuple(outputs.shape), expected_shape
            )
        )


class TestMiniGPT1Shapes(unittest.TestCase):
    def setUp(self):
        self.vocabulary_size = 11
        self.num_heads = 13
        self.embedding_size = 17 * self.num_heads
        self.sequence_length = 23
        self.num_layers = 3

        self.model = MiniGPT1(
            self.vocabulary_size,
            self.embedding_size,
            self.sequence_length,
            self.num_heads,
            self.num_layers,
        )

    @weight(0)
    def test_forward_shape(self):
        batch_size = 7
        tokens = torch.randint(
            self.vocabulary_size, size=(batch_size, self.sequence_length)
        )

        log_probas = self.model(tokens)

        expected_shape = (batch_size, self.sequence_length, self.vocabulary_size)
        assert isinstance(log_probas, torch.Tensor), (
            "The output of the module " "must be a torch.Tensor."
        )
        assert log_probas.shape == expected_shape, (
            "The shape of the output of "
            "the module is invalid with `tokens` of shape {0}.\n  Got shape {1}"
            "\n  Expected shape: {2}\nRecall that the output should have shape "
            "(batch_size, sequence_length, vocabulary_size)".format(
                tuple(tokens.shape), tuple(log_probas.shape), expected_shape
            )
        )

    @weight(0)
    def test_loss_shape(self):
        batch_size = 7
        log_probas = torch.rand(batch_size, self.sequence_length, self.vocabulary_size)
        targets = torch.randint(
            self.vocabulary_size, size=(batch_size, self.sequence_length)
        )
        mask = torch.ones((batch_size, self.sequence_length), dtype=torch.float32)

        loss = self.model.loss(log_probas, targets, mask)

        assert isinstance(loss, torch.Tensor), (
            "The output of `loss` must be " "a torch.Tensor."
        )
        assert loss.shape == (), (
            "The shape of the output of `loss` is invalid "
            "with `log_probas` of shape {0}, `targets` of shape {1}, and `mask` "
            "of shape {2}.\n  Got shape {3}\n  Expected shape: ()\nRecall that "
            "the output of the loss function must be scalar.".format(
                tuple(log_probas.shape),
                tuple(targets.shape),
                tuple(mask.shape),
                tuple(loss.shape),
            )
        )
