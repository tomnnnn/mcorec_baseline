Baseline evaluation of effectivness of using combination of segments overlaps and LLMs for speaker clustering task.

# Task

Given a pair of speaker segments, the LLM is tasked to provide a number from 0 to 1 based on how likely they are to be engaged in the same conversation. This score is averaged with score provided in the baseline, which is calculated based on the segments overlap.

# Metrics
Avg. F1 accross all sessions.
