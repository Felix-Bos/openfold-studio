"""Validation of user input coming from the HTML forms."""

from django import forms

from .domain.attention import LAYER_FAMILY_CHOICES, LAYER_FAMILY_KEYS
from .domain.sequence import parse_sequence
from .errors import InvalidSequenceError


class PredictionForm(forms.Form):
    sequence = forms.CharField(
        label="Protein sequence (one-letter code)",
        widget=forms.Textarea,
        error_messages={"required": "Please enter a protein sequence."},
    )

    def clean_sequence(self) -> str:
        try:
            return parse_sequence(self.cleaned_data["sequence"])
        except InvalidSequenceError as error:
            raise forms.ValidationError(error.message) from error


class AttentionRunForm(forms.Form):
    families = forms.MultipleChoiceField(choices=LAYER_FAMILY_CHOICES, required=False)

    def clean_families(self) -> list[str]:
        """No family ticked means "all families"."""
        return self.cleaned_data["families"] or list(LAYER_FAMILY_KEYS)
