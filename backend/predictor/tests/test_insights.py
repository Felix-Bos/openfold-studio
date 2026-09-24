import numpy as np
from django.test import SimpleTestCase

from predictor.domain.attention import block_statistics, contact_precision, top_long_range_links
from predictor.domain.confidence import (
    band_fractions,
    downsample_matrix,
    low_confidence_regions,
    verdict,
)
from predictor.domain.structure import ca_distance_matrix, parse_residues

from .factories import CIF


class StructureTests(SimpleTestCase):
    def test_parses_one_residue_per_alpha_carbon(self):
        residues = parse_residues(CIF)
        self.assertEqual([r.index for r in residues], [1, 2, 3, 4])
        self.assertEqual(residues[0].plddt, 91.0)
        self.assertEqual(residues[0].name, "MET")

    def test_distance_matrix(self):
        distances = ca_distance_matrix(parse_residues(CIF))
        self.assertAlmostEqual(float(distances[0, 3]), 15.0, places=4)


class ConfidenceTests(SimpleTestCase):
    def test_band_fractions(self):
        fractions = band_fractions([95, 80, 60, 40])
        self.assertEqual(fractions, {"very_high": 0.25, "confident": 0.25, "low": 0.25, "very_low": 0.25})

    def test_low_confidence_regions_need_three_residues(self):
        plddt = [90, 60, 60, 90, 50, 40, 30, 20, 90]
        self.assertEqual(
            [(r["start"], r["end"]) for r in low_confidence_regions(plddt)],
            [(5, 8)],
        )

    def test_verdict_mentions_fold_quality(self):
        self.assertIn("likely correct", verdict(92.0, 0.9))
        self.assertIn("disordered", verdict(30.0, 0.2))

    def test_downsample_keeps_small_matrices(self):
        matrix = np.ones((10, 10))
        self.assertIs(downsample_matrix(matrix, 256), matrix)
        self.assertEqual(downsample_matrix(np.ones((300, 300)), 100).shape, (100, 100))


class AttentionMetricsTests(SimpleTestCase):
    def test_masses_split_by_sequence_separation(self):
        matrix = np.eye(30)  # pure self-attention
        stats = block_statistics(matrix)
        self.assertAlmostEqual(stats["diagonal_mass"], 1.0)
        self.assertAlmostEqual(stats["long_range_mass"], 0.0)
        self.assertAlmostEqual(stats["entropy"], 0.0, places=5)

    def test_top_links_are_long_range_and_checked_against_contacts(self):
        matrix = np.zeros((30, 30))
        matrix[0, 29] = 1.0  # 29 residues apart
        matrix[3, 4] = 5.0  # strong but local: ignored
        distances = np.full((30, 30), 20.0)
        distances[0, 29] = 6.0
        links = top_long_range_links(matrix, distances, count=1)
        self.assertEqual((links[0]["i"], links[0]["j"]), (0, 29))
        self.assertTrue(links[0]["is_contact"])
        self.assertEqual(contact_precision(links), 1.0)


class AttentionSinkTests(SimpleTestCase):
    def test_detects_a_column_everyone_attends_to(self):
        matrix = np.full((40, 40), 0.1)
        matrix[:, 7] = 10.0
        from predictor.domain.attention import attention_sinks

        sinks = attention_sinks(matrix)
        self.assertEqual(sinks[0]["index"], 7)
        self.assertEqual(len(sinks), 1)

    def test_uniform_attention_has_no_sink(self):
        from predictor.domain.attention import attention_sinks

        self.assertEqual(attention_sinks(np.ones((30, 30))), [])
