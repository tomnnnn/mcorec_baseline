"""
Modified evaluate.py script to only evaluate speaker clustering performance.

@author: Hai Phong Nguyen
@date: November 2025
"""
import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import jiwer
import webvtt
import json
from src.cluster.conv_spks import (
    get_clustering_f1_score,
    get_speaker_clustering_f1_score
)
from src.tokenizer.norm_text import remove_disfluencies
import glob
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
text_normalizer = EnglishTextNormalizer({})

def evaluate_conversation_clustering(label_path, output_path):
    with open(os.path.join(label_path, "speaker_to_cluster.json"), "r") as f:
        label_data = json.load(f)
    with open(os.path.join(output_path, "speaker_to_cluster.json"), "r") as f:
        output_data = json.load(f)
    return get_clustering_f1_score(label_data, output_data)

def evaluate_speaker_clustering(label_path, output_path):
    with open(os.path.join(label_path, "speaker_to_cluster.json"), "r") as f:
        label_data = json.load(f)
    with open(os.path.join(output_path, "speaker_to_cluster.json"), "r") as f:
        output_data = json.load(f)
    return get_speaker_clustering_f1_score(label_data, output_data)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate speaker clustering and transcripts from video")
    parser.add_argument('--session_dir', type=str, required=True, help='Path to folder containing session data')
    parser.add_argument('--output_dir_name', type=str, default='output', help='Name of the output directory within each session (default: output)')
    parser.add_argument('--label_dir_name', type=str, default='labels', help='Name of the label directory within each session (default: labels)')
    opt = parser.parse_args()

    if opt.session_dir.strip().endswith("*"):
        all_session_dirs = glob.glob(opt.session_dir)
    else:
        all_session_dirs = [opt.session_dir]
    print(f"Evaluating {len(all_session_dirs)} sessions")

    all_conversation_clustering_f1_score = []

    for session_dir in all_session_dirs:
        print(f"Evaluating session {session_dir.split('/')[-1]}")
        label_path = os.path.join(session_dir, opt.label_dir_name)
        output_path = os.path.join(session_dir, opt.output_dir_name)
        assert os.path.exists(label_path), f"Label path {label_path} does not exist"
        assert os.path.exists(output_path), f"Output path {output_path} does not exist"

        conversation_clustering_f1_score = evaluate_conversation_clustering(label_path, output_path)
        print(f"Conversation clustering F1 score: {conversation_clustering_f1_score}")
        all_conversation_clustering_f1_score.append(conversation_clustering_f1_score)

        speaker_clustering_f1_score = evaluate_speaker_clustering(label_path, output_path)
        print(f"Speaker clustering F1 score: {speaker_clustering_f1_score}")

    print(f"Average Conversation Clustering F1 score: {sum(all_conversation_clustering_f1_score) / len(all_conversation_clustering_f1_score)}")

if __name__ == "__main__":
    main()
