"""Protein sequence normalisation and validation."""

import re

from predictor.errors import InvalidSequenceError

# The 20 standard amino acids plus the IUPAC ambiguity/rare codes
# (X unknown, B/Z/J ambiguous, U selenocysteine, O pyrrolysine).
VALID_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYXBZJUO")

_WHITESPACE_RE = re.compile(r"\s+")

# Well-studied proteins offered as one-click examples on the home page.
EXAMPLE_SEQUENCES = (
    {
        "name": "Ubiquitin",
        "organism": "Human · 76 aa",
        "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    },
    {
        "name": "Lysozyme C",
        "organism": "Hen egg white · 129 aa",
        "sequence": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    },
    {
        "name": "Hemoglobin α",
        "organism": "Human · 142 aa",
        "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    },
    {
        "name": "Green fluorescent protein",
        "organism": "Aequorea victoria · 238 aa",
        "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    },
)


def normalize_sequence(raw: str) -> str:
    """Removes all whitespace (line breaks from a pasted FASTA body) and upper-cases."""
    return _WHITESPACE_RE.sub("", raw or "").upper()


def validate_sequence(sequence: str) -> None:
    """Raises InvalidSequenceError unless `sequence` is a non-empty one-letter sequence."""
    if not sequence:
        raise InvalidSequenceError("Please enter a protein sequence.")

    invalid = sorted(set(sequence) - VALID_RESIDUES)
    if invalid:
        raise InvalidSequenceError(
            "Invalid sequence: only one-letter amino-acid codes are accepted "
            f"(rejected characters: {', '.join(invalid)})."
        )


def parse_sequence(raw: str) -> str:
    """Normalises then validates user input; returns the clean sequence."""
    sequence = normalize_sequence(raw)
    validate_sequence(sequence)
    return sequence
