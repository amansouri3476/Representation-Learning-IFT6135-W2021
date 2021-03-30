import unittest
import torch

from gradescope_utils.autograder_utils.decorators import weight

from lstm_solution import LSTM


class TestLSTMShapes(unittest.TestCase):
    def setUp(self):
        self.vocabulary_size = 11
        self.embedding_size = 13
        self.hidden_size = 17
        self.num_layers = 3

        self.model = LSTM(
            self.vocabulary_size, self.embedding_size, self.hidden_size, self.num_layers
        )

    @weight(0)
    def test_forward_shape(self):
        batch_size = 5
        sequence_length = 7
        tokens = torch.randint(self.vocabulary_size, size=(batch_size, sequence_length))
        initial_states = self.model.initial_states(batch_size)

        outputs = self.model(tokens, initial_states)

        assert isinstance(outputs, (tuple, list)), (
            "The output of the module " "must be a tuple (log_probas, hidden_states)."
        )
        assert len(outputs) == 2, (
            "The output of the module must be a tuple " "(log_probas, hidden_states)."
        )

        log_probas, hidden_states = outputs

        expected_shape = (batch_size, sequence_length, self.vocabulary_size)
        assert isinstance(log_probas, torch.Tensor), (
            "The output `log_probas` " "must be a torch.Tensor."
        )
        assert log_probas.shape == expected_shape, (
            "The shape of `log_probas` "
            "is invalid with `tokens` of shape {0}.\n  Got shape {1}"
            "\n  Expected shape: {2}\nRecall that the output should have shape "
            "(batch_size, sequence_length, vocabulary_size)".format(
                tuple(tokens.shape), tuple(log_probas.shape), expected_shape
            )
        )

        expected_shape = (self.num_layers, batch_size, self.hidden_size)
        assert isinstance(hidden_states, (tuple, list)), (
            "The output " "`hidden_states` must ba a tuple (h, c)."
        )
        assert len(hidden_states) == 2, (
            "The output `hidden_states` must ba a " "tuple (h, c)."
        )
        assert hidden_states[0].shape == expected_shape, (
            "The shape of `h` is "
            "invalid with `tokens` of shape {0}.\n  Got shape {1}"
            "\n  Expected shape: {2}\nRecall that the output should have shape "
            "(num_layers, batch_size, hidden_size)".format(
                tuple(tokens.shape), tuple(log_probas.shape), expected_shape
            )
        )
        assert hidden_states[1].shape == expected_shape, (
            "The shape of `c` is "
            "invalid with `tokens` of shape {0}.\n  Got shape {1}"
            "\n  Expected shape: {2}\nRecall that the output should have shape "
            "(num_layers, batch_size, hidden_size)".format(
                tuple(tokens.shape), tuple(log_probas.shape), expected_shape
            )
        )

    @weight(0)
    def test_loss_shape(self):
        batch_size = 5
        sequence_length = 7
        log_probas = torch.rand(batch_size, sequence_length, self.vocabulary_size)
        targets = torch.randint(
            self.vocabulary_size, size=(batch_size, sequence_length)
        )
        mask = torch.ones((batch_size, sequence_length), dtype=torch.float32)

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
