"""Object recognition by voting columns: soundness, exact elimination rates, convergence."""

import math
from fractions import Fraction
from itertools import pairwise

import pytest
import torch

from neurosush.htm.objects import ColumnEnsemble, ObjectLibrary, SensorColumn, vote


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def stirling2(n, k):
    """Ways to split ``n`` labelled items into ``k`` non-empty groups."""
    return sum((-1) ** j * math.comb(k, j) * (k - j) ** n for j in range(k + 1)) // math.factorial(
        k
    )


def distinct_moment(locations, features, power):
    """``E[(D / features)^power]`` for ``D`` distinct values among uniform draws."""
    return sum(
        Fraction(
            math.comb(features, d) * math.factorial(d) * stirling2(locations, d),
            features**locations,
        )
        * Fraction(d, features) ** power
        for d in range(1, min(locations, features) + 1)
    )


def distinct_locations(shape, count, g):
    order = torch.randperm(shape[0] * shape[1], generator=g)[:count]
    return [(int(i) // shape[1], int(i) % shape[1]) for i in order]


def touch(ensemble, library, obj, locations):
    return ensemble.sense([int(library.features[obj, y, x]) for y, x in locations])


class TestColumn:
    def test_sensing_keeps_matching_locations(self):
        library = ObjectLibrary(torch.tensor([[[1, 2], [2, 3]], [[3, 3], [1, 0]]]))
        column = SensorColumn(library)
        column.sense(2)
        assert column.hypotheses.nonzero().tolist() == [[0, 0, 1], [0, 1, 0]]
        assert column.candidates().tolist() == [True, False]

    def test_hypotheses_move_with_the_sensor(self):
        library = ObjectLibrary(torch.arange(9).reshape(1, 3, 3))
        column = SensorColumn(library)
        column.sense(4)  # the center
        column.move((1, -1))
        assert column.hypotheses[0].nonzero().tolist() == [[2, 0]]
        column.move((1, 0))  # wraps around like a grid cell phase
        assert column.hypotheses[0].nonzero().tolist() == [[0, 0]]
        column.sense(0)
        assert column.candidates().tolist() == [True]

    def test_true_hypothesis_is_never_lost(self):
        g = gen()
        library = ObjectLibrary.random(30, (4, 4), 5, generator=g)
        for obj in range(30):
            column = SensorColumn(library)
            y, x = 1, 2
            for _ in range(10):
                column.sense(int(library.features[obj, y, x]))
                assert column.hypotheses[obj, y, x]
                dy, dx = torch.randint(-1, 2, (2,), generator=g).tolist()
                column.move((dy, dx))
                y, x = (y + dy) % 4, (x + dx) % 4

    def test_single_column_false_hypotheses_decay_as_k_to_the_minus_t(self):
        # a hypothesis on another object survives t sensations at distinct locations with
        # probability exactly K^-t, so E[survivors] = (N - 1) * locations * K^-t
        objects, shape, features, trials = 10, (4, 4), 6, 3000
        g = gen(1)
        survivors = torch.zeros(trials, 3)
        for trial in range(trials):
            library = ObjectLibrary.random(objects, shape, features, generator=g)
            column = SensorColumn(library)
            path = distinct_locations(shape, 3, g)
            for t, (y, x) in enumerate(path):
                if t:
                    py, px = path[t - 1]
                    column.move((y - py, x - px))
                column.sense(int(library.features[0, y, x]))
                survivors[trial, t] = column.hypotheses[1:].sum()
        expected = torch.tensor([(objects - 1) * 16 * features ** -(t + 1) for t in range(3)])
        error = survivors.std(0) / math.sqrt(trials)
        assert ((survivors.mean(0) - expected).abs() < 5 * error).all()


class TestVoting:
    def test_vote_counts_supporting_columns(self):
        candidates = torch.tensor([[True, True, False], [True, False, False], [True, True, True]])
        assert vote(candidates).tolist() == [True, False, False]
        assert vote(candidates, min_support=2).tolist() == [True, True, False]
        with pytest.raises(ValueError, match="min_support"):
            vote(candidates, min_support=4)

    @pytest.mark.parametrize("columns", [1, 2, 3])
    def test_one_sensation_matches_the_occupancy_formula(self, columns):
        # a wrong object stays a candidate of a column iff the sensed feature occurs on it;
        # given its D distinct features the columns are independent, so it survives the
        # vote with probability E[(D / K)^C]
        objects, shape, features, trials = 20, (3, 3), 20, 3000
        g = gen(columns)
        wrong = torch.zeros(trials)
        for trial in range(trials):
            library = ObjectLibrary.random(objects, shape, features, generator=g)
            ensemble = ColumnEnsemble(library, columns)
            consensus = touch(ensemble, library, 0, distinct_locations(shape, columns, g))
            assert consensus[0]
            wrong[trial] = consensus[1:].sum()
        expected = (objects - 1) * float(distinct_moment(9, features, columns))
        assert abs(wrong.mean().item() - expected) < 5 * wrong.std().item() / math.sqrt(trials)

    def test_more_columns_recognize_in_fewer_sensations(self):
        # Lewis et al. (2019), Fig. 5: voting cuts the sensations needed
        objects, shape, features = 100, (5, 5), 30
        library = ObjectLibrary.random(objects, shape, features, generator=gen(7))
        g = gen(8)
        means = []
        for columns in (1, 2, 4, 8):
            ensemble = ColumnEnsemble(library, columns)
            counts = []
            for obj in range(0, objects, 2):
                ensemble.reset()
                locations = distinct_locations(shape, columns, g)
                for step in range(1, 30):  # noqa: B007 (read after the loop)
                    touch(ensemble, library, obj, locations)
                    if ensemble.recognized() is not None:
                        break
                    moves = torch.randint(-1, 2, (columns, 2), generator=g).tolist()
                    ensemble.move([tuple(m) for m in moves])
                    locations = [
                        ((y + dy) % 5, (x + dx) % 5)
                        for (y, x), (dy, dx) in zip(locations, moves, strict=True)
                    ]
                assert ensemble.recognized() == obj
                counts.append(step)
            means.append(sum(counts) / len(counts))
        assert all(b < a for a, b in pairwise(means))
        assert means[0] > 3 > 2 > means[-1]  # measured: 3.6, 2.26, 2.0, 1.72

    def test_min_support_tolerates_a_faulty_column(self):
        library = ObjectLibrary(
            torch.tensor([[[0, 1], [2, 3]], [[0, 1], [2, 4]], [[5, 6], [7, 8]]])
        )
        strict, tolerant = ColumnEnsemble(library, 3), ColumnEnsemble(library, 3, min_support=2)
        sensed = [0, 1, 9]  # the third column misreads its feature
        assert not strict.sense(sensed).any()
        assert tolerant.sense(sensed).tolist() == [True, True, False]


class TestValidation:
    def test_library_checks(self):
        with pytest.raises(ValueError, match="integer"):
            ObjectLibrary(torch.zeros(2, 2, 2))
        with pytest.raises(ValueError, match="non-negative"):
            ObjectLibrary(-torch.ones(2, 2, 2, dtype=torch.long))

    def test_ensemble_checks(self):
        library = ObjectLibrary(torch.zeros(1, 2, 2, dtype=torch.long))
        with pytest.raises(ValueError, match="columns"):
            ColumnEnsemble(library, 0)
        with pytest.raises(ValueError, match="expected 2 features"):
            ColumnEnsemble(library, 2).sense([0])
