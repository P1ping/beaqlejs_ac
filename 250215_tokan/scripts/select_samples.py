import os
import json
import random
import argparse
import numpy as np

from collections import defaultdict

ACCENT_TO_SPEAKER = {
    "<ar>": ["ABA", "YBAA", "ZHAA", "SKA"],
    "<zh>": ["BWC", "LXC", "NCC", "TXHC"],
    "<hi>": ["ASI", "RRBI", "SVBI", "TNI"],
    "<ko>": ["HJK", "YDCK", "YKWK", "HKK"],
    "<es>": ["EBVS", "ERMS", "NJS", "MBMPS"],
    "<vi>": ["HQTV", "PNV", "THV", "TLV"],
    "<us>": ["BDL", "RMS", "SLT", "CLB"],
}
SPEAKER_TO_ACCENT = {
    spk: accent for accent, spks in ACCENT_TO_SPEAKER.items() for spk in spks
}
SPEAKERS = list(SPEAKER_TO_ACCENT.keys())


NATURALNESS_SAMPLES = 650
ACCENTEDNESS_SAMPLES = 600
SPKSIM_SAMPLES = 650


ACCENT_EXCLUDED_SPEAKERS = {"BDL", "RMS", "SLT", "CLB"}


def load_test_file(test_file):
    spk2sids = defaultdict(set)
    sid2text = {}
    with open(test_file, "r") as f:
        for line in f:
            uid, text = line.strip().split("\t")
            spk, sid = uid.split("_", maxsplit=1)
            spk2sids[spk].add(sid)
            if sid not in sid2text:
                sid2text[sid] = text
    return spk2sids, sid2text


def get_speaker_weights():
    spk2weights = {}
    all_speakers = set(SPEAKERS)
    spk2weights = {spk: 1 / len(all_speakers) for spk in all_speakers}
    return spk2weights


def get_speaker_num_samples(num_samples, spk2weights):
    spk2num_samples = {spk: int(w * num_samples) for spk, w in spk2weights.items()}
    total_samples = sum(n for n in spk2num_samples.values())

    num_residual = num_samples - total_samples
    assert num_residual >= 0

    if num_residual > 0:
        residual_choices = random.choices(
            list(spk2weights.keys()), k=num_residual, weights=spk2num_samples.values()
        )
        for spk in residual_choices:
            spk2num_samples[spk] += 1

    return spk2num_samples


def select_samples(spk2num_samples, spk2sids):
    assert len(spk2num_samples) == len(spk2sids)
    sample_list = []
    for spk, num_samples in spk2num_samples.items():
        assert num_samples <= len(spk2sids[spk])
        sid_group = random.sample(spk2sids[spk], k=num_samples)
        for sid in sid_group:
            sample_list.append((spk, sid))
            spk2sids[spk].remove(sid)
    return sample_list, spk2sids


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=777)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    spk2sids, sid2text = load_test_file(args.test_file)

    spk2weights = get_speaker_weights()

    num_samples = NATURALNESS_SAMPLES + ACCENTEDNESS_SAMPLES + SPKSIM_SAMPLES

    spk2num_samples = get_speaker_num_samples(NATURALNESS_SAMPLES, spk2weights)
    naturalness_samples, spk2sids = select_samples(spk2num_samples, spk2sids)

    tmp_spk2weights = spk2weights.copy()
    for spk in ACCENT_EXCLUDED_SPEAKERS:
        tmp_spk2weights[spk] = 0
    spk2num_samples = get_speaker_num_samples(ACCENTEDNESS_SAMPLES, tmp_spk2weights)
    accentedness_samples, spk2sids = select_samples(spk2num_samples, spk2sids)

    spk2num_samples = get_speaker_num_samples(SPKSIM_SAMPLES, spk2weights)
    spksim_samples, spk2sids = select_samples(spk2num_samples, spk2sids)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "naturalness.json"), "w") as f:
        naturalness_metadata = [
            {"speaker": spk, "sentence_id": sid, "text": sid2text[sid]}
            for spk, sid in naturalness_samples
        ]
        json.dump(naturalness_metadata, f, indent=4)

    with open(os.path.join(args.output_dir, "accentedness.json"), "w") as f:
        accentedness_metadata = [
            {"speaker": spk, "sentence_id": sid, "text": sid2text[sid]}
            for spk, sid in accentedness_samples
        ]
        json.dump(accentedness_metadata, f, indent=4)

    with open(os.path.join(args.output_dir, "spksim.json"), "w") as f:
        spksim_metadata = [
            {"speaker": spk, "sentence_id": sid, "text": sid2text[sid]}
            for spk, sid in spksim_samples
        ]
        json.dump(spksim_metadata, f, indent=4)
