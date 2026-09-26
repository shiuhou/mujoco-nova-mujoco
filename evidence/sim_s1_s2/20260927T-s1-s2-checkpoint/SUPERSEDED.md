# Superseded evidence record

This first generated S1/S2 benchmark record is retained to preserve research
history. Its per-case `action_frame` field was mislabeled as `link6` even
though Cartesian positions were interpreted in the MuJoCo `base` frame.

The top-level fields correctly recorded `action_frame: base` and
`controlled_frame: link6`; the numerical benchmark results are unaffected.
Use the later `20260927T-s1-s2-checkpoint-002` record for the corrected schema
and final checkpoint results.
