import pytest
import torch

from neurosush.readout import accuracy, assign_labels, classify


def counts():
    """Four samples (classes 0, 0, 1, 2) and five neurons."""
    return torch.tensor(
        [
            [4.0, 0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 5.0, 3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 6.0, 0.0],
        ]
    )


class TestAssignLabels:
    def test_each_neuron_takes_the_class_with_the_highest_mean(self):
        # means per class: 0 -> [3, 0, 2, 0, 0], 1 -> [0, 5, 3, 0, 0], 2 -> [0, 1, 0, 6, 0]
        assignment = assign_labels(counts(), torch.tensor([0, 0, 1, 2]), classes=3)
        assert assignment.tolist() == [0, 1, 1, 2, -1]

    def test_the_mean_is_per_class_not_the_sum(self):
        # class 0 has two samples with 2 spikes each (sum 4), class 1 one sample with 3
        counts = torch.tensor([[2.0], [2.0], [3.0]])
        assert assign_labels(counts, torch.tensor([0, 0, 1]), classes=2).tolist() == [1]

    def test_a_tie_goes_to_the_lowest_class(self):
        counts = torch.tensor([[2.0], [2.0]])
        assert assign_labels(counts, torch.tensor([0, 1]), classes=2).tolist() == [0]

    def test_neurons_that_never_respond_are_unassigned(self):
        assignment = assign_labels(torch.zeros(3, 2), torch.tensor([0, 1, 2]), classes=3)
        assert assignment.tolist() == [-1, -1]

    def test_classes_without_samples_have_zero_response(self):
        counts = torch.tensor([[1.0, 0.0]])
        assert assign_labels(counts, torch.tensor([2]), classes=4).tolist() == [2, -1]

    def test_integer_counts_work(self):
        counts = torch.tensor([[1, 0], [0, 2]])
        assert assign_labels(counts, torch.tensor([0, 1]), classes=2).tolist() == [0, 1]

    @pytest.mark.parametrize(
        ("labels", "match"),
        [(torch.tensor([0, 1, 2]), "shape"), (torch.tensor([0, 1, 2, 3]), r"\[0, 3\)")],
    )
    def test_invalid_labels(self, labels, match):
        with pytest.raises(ValueError, match=match):
            assign_labels(counts(), labels, classes=3)

    def test_counts_must_be_two_dimensional(self):
        with pytest.raises(ValueError, match="samples, neurons"):
            assign_labels(torch.zeros(3), torch.zeros(3, dtype=torch.long))


class TestClassify:
    def test_the_class_with_the_highest_mean_over_its_neurons_wins(self):
        assignment = torch.tensor([0, 1, 1, 2, -1])
        # sample 0: class 0 -> 4, class 1 -> (0 + 1) / 2, class 2 -> 0
        # sample 1: class 0 -> 2, class 1 -> (0 + 3) / 2, class 2 -> 0
        # sample 2: class 0 -> 0, class 1 -> (5 + 3) / 2, class 2 -> 0
        # sample 3: class 0 -> 0, class 1 -> (1 + 0) / 2, class 2 -> 6
        got = classify(counts(), assignment, classes=3)
        assert got.tolist() == [0, 0, 1, 2]

    def test_the_mean_means_a_big_group_has_no_advantage(self):
        counts = torch.tensor([[3.0, 3.0, 3.0, 4.0]])
        assignment = torch.tensor([0, 0, 0, 1])
        assert classify(counts, assignment, classes=2).tolist() == [1]  # 4 > 3, not 9 > 4

    def test_unassigned_neurons_are_ignored(self):
        counts = torch.tensor([[100.0, 1.0]])
        assert classify(counts, torch.tensor([-1, 1]), classes=2).tolist() == [1]

    def test_classes_without_neurons_cannot_win(self):
        counts = torch.tensor([[0.0, 0.0]])  # class 0 has no neuron; class 1 scores 0
        assert classify(counts, torch.tensor([1, 1]), classes=2).tolist() == [1]

    def test_with_no_assigned_neuron_every_class_scores_minus_infinity(self):
        assert classify(torch.ones(2, 2), torch.tensor([-1, -1]), classes=3).tolist() == [0, 0]

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="assignment"):
            classify(counts(), torch.tensor([0, 1]), classes=3)

    def test_a_lone_class_with_neurons_beats_empty_ones(self):
        got = classify(torch.tensor([[1.0]]), torch.tensor([2]), classes=3)
        assert got.tolist() == [2]


class TestAccuracy:
    def test_fraction_correct(self):
        assert accuracy(torch.tensor([0, 1, 2, 2]), torch.tensor([0, 1, 1, 2])) == 0.75

    def test_shapes_must_match(self):
        with pytest.raises(ValueError, match="match"):
            accuracy(torch.tensor([0, 1]), torch.tensor([0]))

    def test_needs_samples(self):
        with pytest.raises(ValueError, match="at least one"):
            accuracy(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
