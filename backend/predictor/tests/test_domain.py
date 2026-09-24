import numpy as np
from django.test import SimpleTestCase

from predictor.domain.attention import (
    get_layer_family,
    parse_threshold,
    select_block,
    sparse_cube_points,
)
from predictor.domain.sequence import normalize_sequence, parse_sequence
from predictor.errors import InvalidSequenceError, ResourceNotFoundError


class SequenceTests(SimpleTestCase):
    def test_normalize_removes_whitespace_and_uppercases(self):
        self.assertEqual(normalize_sequence(" mkt\nay ia\t"), "MKTAYIA")

    def test_parse_accepts_valid_sequence(self):
        self.assertEqual(parse_sequence("mktayiakqr"), "MKTAYIAKQR")

    def test_parse_rejects_empty_sequence(self):
        with self.assertRaises(InvalidSequenceError):
            parse_sequence("   ")

    def test_parse_rejects_invalid_letters_and_lists_them(self):
        with self.assertRaisesMessage(InvalidSequenceError, "1, @"):
            parse_sequence("MKT1A@")


class AttentionDomainTests(SimpleTestCase):
    def test_unknown_family_is_not_found(self):
        with self.assertRaises(ResourceNotFoundError):
            get_layer_family("nope")

    def test_select_block_checks_range(self):
        weights = np.zeros((2, 3, 3))
        self.assertEqual(select_block(weights, 1).shape, (3, 3))
        with self.assertRaises(ResourceNotFoundError):
            select_block(weights, 2)

    def test_parse_threshold_clamps_and_defaults(self):
        self.assertEqual(parse_threshold("1.7"), 1.0)
        self.assertEqual(parse_threshold("-1"), 0.0)
        self.assertEqual(parse_threshold("abc"), 0.6)
        self.assertEqual(parse_threshold(None), 0.6)

    def test_sparse_cube_keeps_points_above_relative_threshold(self):
        cube = np.zeros((2, 2, 2))
        cube[0, 1, 1] = 4.0
        cube[1, 0, 0] = 2.0
        cube[1, 1, 1] = 1.0
        points = sparse_cube_points(cube, threshold=0.5)
        self.assertCountEqual(points, [[0, 1, 1, 1.0], [1, 0, 0, 0.5]])

    def test_sparse_cube_caps_to_strongest_points(self):
        cube = np.arange(8, dtype=float).reshape(2, 2, 2)
        points = sparse_cube_points(cube, threshold=0.0, max_points=2)
        self.assertEqual([round(p[3], 4) for p in sorted(points, key=lambda p: p[3])], [round(6 / 7, 4), 1.0])
