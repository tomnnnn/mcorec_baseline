"""
Uses LLM to cluster speaker segments loaded from {session_dir}/labels folder.
"""

import os
import sys
import json
import argparse
import tqdm
from typing import Dict, Tuple
import numpy as np
import vllm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.cluster.llm_cluster import calculate_conversation_scores_llm
from src.cluster.conv_spks import (cluster_speakers, calculate_conversation_scores)

class ClusterEngine():
    def __init__(self, model: str, num_gpus: int, combined: bool = False):
        self.model = model
        self.num_gpus = num_gpus
        self.combined = combined

    def init_model(self):
        """
        Initialize the LLM model via vllm interface.
        """

        self.llm = vllm.LLM(
            model=self.model,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=self.num_gpus,
            max_model_len=10*1024,
        )

    def __parse_vtt_segments(self, vtt_lines: list) -> list[Tuple[Tuple[float,float], str]]:
        """
        Parse VTT lines to extract segments with timestamps and text.

        Args:
            vtt_lines (list): List of lines from a VTT file.
        Returns:
            list of tuples: Each tuple contains ((start_time, end_time), text).
        """
        segments = []
        start_time, end_time = 0.0, 0.0

        for line in vtt_lines:
            line = line.strip()
            if line.startswith("NOTE") or line.startswith("WEBVTT"):
                continue

            if " --> " in line:
                time_range = line.split(" --> ")
                start_time = float(time_range[0].replace(":", "").replace(",", "."))
                end_time = float(time_range[1].replace(":", "").replace(",", "."))
            elif line:
                text = line
                segments.append(((start_time, end_time), text))

        return segments


    def cluster_session(self, session_dir, output_dir):
        """
        Cluster speaker segments. Stores the clustering results in the speaker_to_cluster.json file inside
        the output directory.

        Args:
            session_dir (str): Path to the session directory containing labels.
            output_dir (str): Path to the output directory to save results.
        """
        labels_dir = os.path.join(session_dir, "labels")
        speaker_files = [f for f in os.listdir(labels_dir) if f.endswith(".vtt")]
        print(f"Found {len(speaker_files)} speaker files in {labels_dir}.")
        speaker_segments: Dict[str, list] = {}
        speaker_ids = []

        for speaker_file in speaker_files:
            speaker_id = os.path.splitext(speaker_file)[0]

            with open(os.path.join(labels_dir, speaker_file), "r") as f:
                vtt_lines = f.readlines()

            segments = self.__parse_vtt_segments(vtt_lines)
            speaker_segments[speaker_id] = segments
            speaker_ids.append(speaker_id)

        # Perform clustering
        conv_scores_llm = calculate_conversation_scores_llm(speaker_segments, self.llm)

        if self.combined:
            conv_scores_spks = calculate_conversation_scores(speaker_segments)
            combined_scores = np.empty((len(speaker_ids), len(speaker_ids)))
            for spk1 in speaker_ids:
                for spk2 in speaker_ids:
                    score_llm = conv_scores_llm[spk1][spk2]
                    score_spks = conv_scores_spks[spk1][spk2]
                    combined_score = (score_llm + score_spks) / 2.0
                    combined_scores[spk1][spk2] = combined_score
            clustered_speakers = cluster_speakers(combined_scores, speaker_ids)
        else:
            clustered_speakers = cluster_speakers(conv_scores_llm, speaker_ids)

        # Save clustering results
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "speaker_to_cluster.json"), "w") as f:
            json.dump(clustered_speakers, f, indent=4)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data-dir", type=str, default="data-bin/dev", help="Path to the data directory containing session folders.")
    args.add_argument("--output-dir", type=str, default="output", help="Path to the output directory to save clustering results.")
    args.add_argument("--model", type=str, default="TBA", help="LLM model to use for clustering.")
    args.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs on one node to use.")
    args.add_argument("--combined", action="store_true", help="Whether to use combined LLM and segment overlap embedding scores.")
    args = args.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    session_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]

    print(f"Clustering sessions in {args.data_dir} using LLM model: {args.model}")

    engine = ClusterEngine(args.model, args.num_gpus, args.combined)
    engine.init_model()

    for session_dir in tqdm.tqdm(session_dirs, desc="Clustering Sessions", unit="session"):
        session_name = os.path.basename(session_dir)
        output_dir = os.path.join(session_dir, args.output_dir)

        print(f"Clustering session: {session_name}")

        engine.cluster_session(session_dir, output_dir)

    print("Clustering completed for all sessions.")
