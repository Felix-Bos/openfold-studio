from django.test import SimpleTestCase

from predictor.openfold.log_parser import (
    STEP_INFERENCE,
    STEP_MSA,
    STEP_TEMPLATES,
    ErrorCollector,
    Phase,
    is_error_line,
    parse_clock,
    parse_line,
)


class LogParserTests(SimpleTestCase):
    def test_parse_clock(self):
        self.assertEqual(parse_clock("01:05"), 65.0)
        self.assertEqual(parse_clock("1:00:00"), 3600.0)
        self.assertIsNone(parse_clock("?"))

    def test_msa_progress_bar(self):
        event = parse_line("COMPLETE:  40%|####      | 60/150 [elapsed: 00:02 remaining: 00:03]")
        self.assertEqual(event.step, STEP_MSA)
        self.assertEqual(event.phase, Phase.MSA)
        self.assertEqual(event.progress_percent, 40.0)
        self.assertEqual(event.elapsed_seconds, 2.0)
        self.assertEqual(event.eta_seconds, 3.0)

    def test_inference_progress_bar(self):
        event = parse_line("Predicting DataLoader 0: 100%|##########| 1/1 [08:03<00:00,  0.00it/s]")
        self.assertEqual(event.step, STEP_INFERENCE)
        self.assertEqual(event.phase, Phase.INFERENCE)
        self.assertEqual(event.elapsed_seconds, 483.0)

    def test_templates_step(self):
        self.assertEqual(parse_line("Preprocessing templates for query").step, STEP_TEMPLATES)

    def test_irrelevant_lines(self):
        self.assertIsNone(parse_line(""))
        self.assertIsNone(parse_line("Loading checkpoint"))

    def test_error_lines(self):
        self.assertTrue(is_error_line("Traceback (most recent call last):"))
        self.assertTrue(is_error_line("  Exception: boom"))
        self.assertFalse(is_error_line("All good"))


class ErrorCollectorTests(SimpleTestCase):
    def test_keeps_the_whole_traceback(self):
        collector = ErrorCollector()
        for line in ["Loading model", "Traceback (most recent call last):", '  File "x.py", line 1', "ValueError: bad shape"]:
            collector.feed(line)
        self.assertEqual(collector.lines[0], "Traceback (most recent call last):")
        self.assertEqual(collector.lines[-1], "ValueError: bad shape")
