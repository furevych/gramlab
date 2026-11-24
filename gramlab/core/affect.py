"""core-v1 · layer 0 · affect — the fixed origin for 'power words'.

A published valence/arousal/dominance lexicon (NRC VAD or Warriner norms; tab-separated
`word  valence  arousal  dominance`, values in [0,1] or rescaled) is loaded as a versioned data file.
It is never learned or retrained.  `force(event)` = VAD of the action, modified by status (negated →
neutral, possible → halved distance from neutral) and by the patient (a person as patient weighs the
dominance axis more).  Power words are simply the corner: low valence, high arousal, high dominance.

Without a lexicon file every call returns None and the affect benchmark reports 'skipped'.
"""
from __future__ import annotations

from pathlib import Path

from .records import Event

NEUTRAL = (0.5, 0.5, 0.5)


class Affect:
    def __init__(self, lexicon_path=None):
        self.lex = {}
        self.source = None
        if lexicon_path and Path(lexicon_path).exists():
            self.load(lexicon_path)

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 4 or p[0].lower() in ("word", "term"):
                    continue
                try:
                    v, a, d = float(p[1]), float(p[2]), float(p[3])
                except ValueError:
                    continue
                if max(v, a, d) > 1.5:          # Warriner norms are 1..9
                    v, a, d = ((x - 1) / 8 for x in (v, a, d))
                self.lex[p[0].lower()] = (v, a, d)
        self.source = str(path)

    def word(self, w):
        return self.lex.get(w.lower())

    def force(self, event: Event):
        vad = self.word(event.action)
        if vad is None:
            return None
        v, a, d = vad
        if event.status == "negated":
            return NEUTRAL
        if event.status == "possible":
            v, a, d = (0.5 + (x - 0.5) * 0.5 for x in (v, a, d))
        if event.patient is not None and event.patient.etype == "person":
            d = min(1.0, 0.5 + (d - 0.5) * 1.25)
        return (round(v, 3), round(a, 3), round(d, 3))

    @staticmethod
    def power(vad, v_max=0.35, a_min=0.6, d_min=0.55):
        """True in the power-word corner."""
        return vad is not None and vad[0] <= v_max and vad[1] >= a_min and vad[2] >= d_min
