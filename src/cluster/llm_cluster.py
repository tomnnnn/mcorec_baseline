import numpy as np
from typing import List, Tuple, Dict
from vllm import LLM

def combine_speaker_segments(speaker_segments: Dict[str, List[Tuple[Tuple[float, float], str]]]) -> str:
    """
    Combine all speaker segments into a single formatted string.
    
    Args:
        speaker_segments: Dictionary mapping speaker IDs to their segments. Segment
        contains a list of tuples with (start_time, end_time) and the spoken text.
        
    Returns:
        Combined string of all speaker segments.
    """
    combined_text = ""
    segments = [[speaker, time_range, text] 
                for speaker, segs in speaker_segments.items() 
                for time_range, text in segs]
    segments.sort(key=lambda x: x[1][0])  # Sort by start time

    for speaker, time_range, text in segments:
        start_time = time_range[0]
        combined_text += f"[{start_time:.2f}] {speaker}: {text}\n"

    return combined_text

def calculate_conversation_scores_llm(speaker_segments: Dict[str, List[Tuple[Tuple[float, float], str]]], 
                                      model: LLM) -> np.ndarray:
    """
    Calculate conversation likelihood scores between all pairs of speakers.
    Higher score means speakers are more likely to be in the same conversation.
    Alternative method using LLM to predict conversation likelihood.
    Uses API to access LLM for predictions.
    
    Args:
        speaker_segments: Dictionary mapping speaker IDs to their segments. Segment
        contains a list of tuples with (start_time, end_time) and the spoken text.
        base_url: Base URL for the LLM API.
        api_key: API key for authentication (if required).
        
    Returns:
        NxN numpy array of conversation scores
    """

    scores = np.zeros((len(speaker_segments), len(speaker_segments)))
    speaker_ids = list(speaker_segments.keys())

    for i, speaker_a in enumerate(speaker_ids):
        for j, speaker_b in enumerate(speaker_ids):
            if i >= j:
                continue  # Avoid redundant calculations

            segments_a = speaker_segments[speaker_a]
            segments_b = speaker_segments[speaker_b]

            combined_text = combine_speaker_segments({speaker_a: segments_a, speaker_b: segments_b})

            prompt = (
                f"""
                Conversation segments:
                {combined_text}
                """
            )

            completion = model.chat(messages=[
                    {"role": "system", "content": """
                        You are an evaluator determining whether two sets of utterances come from the same
                        conversation. Often, the segments are from completely unrelated conversations.

                        Your task is to detect whether the segments logically follow each other in topic, 
                        participants, and conversational context. Consider contradictions in topic, tone, time,
                        and speaker roles as strong evidence that they are NOT from the same conversation.

                        Return a score strictly between 0 and 1:
                        - 0 = definitely different conversations
                        - 1 = definitely the same conversation
                        - Outputs near 0.5 should only be used when there is genuine ambiguity.

                        Do NOT assume coherence unless the evidence strongly supports it.
                        Strictly return only the numeric score."""},
                    {"role": "user", "content": prompt}
            ],
            use_tqdm=True)

            output = completion[0].outputs[0].text.strip()
            score = float(output)

            scores[i, j] = score
            scores[j, i] = score  # Symmetric matrix

    return scores
