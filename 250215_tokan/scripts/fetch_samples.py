import os
import json
import argparse
import numpy as np
import torchaudio
from tqdm import tqdm
import shutil
import copy
import soundfile as sf

AUDIO_DIR = "./listening_test_interspeech2025/samples"
OUTPUT_DIR = "./audio/20250215_tokan"

SAMPLES_PER_TEST = {
    "MUSHRANAT": 20,
    "MUSHRAACT": 20,
    "BWSSIM": 20,
}

SYSTEM_INFO = [
    {
        "system": "Reference",
        "audio_dir": f"{AUDIO_DIR}/source",
        "name_pattern": "{spk}_{sid}.wav",
        "BWS_label": "Reference",
    },
    {
        "system": "source",
        "audio_dir": f"{AUDIO_DIR}/source",
        "name_pattern": "{spk}_{sid}.wav",
        "BWS_label": None,
    },
    {
        "system": "baseline",
        "audio_dir": f"{AUDIO_DIR}/baseline",
        "name_pattern": "{spk}_{sid}.wav",
        "BWS_label": "W",
    },
    {
        "system": "ours",
        "audio_dir": f"{AUDIO_DIR}/ours",
        "name_pattern": "{spk}_{sid}.wav",
        "BWS_label": "Z",
    },
    {
        "system": "ours_scaling",
        "audio_dir": f"{AUDIO_DIR}/ours_scaling",
        "name_pattern": "{spk}_{sid}.wav",
        "BWS_label": "Y",
    },
    {
        "system": "ours_control",
        "audio_dir": f"{AUDIO_DIR}/ours_control",
        "name_pattern": "{spk}_{sid}.wav",
        "BWS_label": "X",
    },
]
SYSTEM_INFO_DICT = {item["system"]: item for item in SYSTEM_INFO}

NATURALNESS_SYSTEMS = [
    "source",
    "baseline",
    "ours",
    "ours_scaling",
    "ours_control",
]
ACCENTEDNESS_SYSTEMS = [
    "source",
    "baseline",
    "ours",
    "ours_scaling",
    "ours_control",
]
SPKSIM_SYSTEMS = [
    "Reference",
    "baseline",
    "ours",
    "ours_scaling",
    "ours_control",
]


NATURALNESS_SAMPLE_LIST = (
    "./listening_test_interspeech2025/split_metadata/naturalness.json"
)
ACCENTEDNESS_SAMPLE_LIST = (
    "./listening_test_interspeech2025/split_metadata/accentedness.json"
)
SPKSIM_SAMPLE_LIST = "./listening_test_interspeech2025/split_metadata/spksim.json"


def load_metadata(list_path):
    with open(list_path, "r") as f:
        metadata = json.load(f)
    return metadata


def fetch_samples(metadata, output_dir, system_info_dict, tag):
    output_metadata = []
    for sample in tqdm(metadata):
        spk = sample["speaker"]
        sid = sample["sentence_id"]
        text = sample["text"]
        item = {
            "Name": f"{sid}",
            "TestID": f"{tag}---{sid}---{spk}",
            "Text": text,
        }

        files_dict = {}
        for system, system_info in system_info_dict.items():
            orig_name = system_info["name_pattern"].format(spk=spk, sid=sid)
            orig_path = os.path.join(system_info["audio_dir"], orig_name)
            targ_dir = os.path.join(output_dir, system, spk)
            os.makedirs(targ_dir, exist_ok=True)
            targ_path = os.path.join(targ_dir, f"{tag}-{sid}.wav")
            resample_and_copy(orig_path, targ_path)
            files_dict[system] = targ_path

        item["Files"] = files_dict
        output_metadata.append(item)

    return output_metadata


def resample_and_copy(orig_path, targ_path, targ_sr=22050):
    with sf.SoundFile(orig_path) as f:
        sr = f.samplerate

    if sr == targ_sr:
        shutil.copy(orig_path, targ_path)
        return

    wav, orig_sr = torchaudio.load(orig_path)
    wav = torchaudio.functional.resample(
        wav,
        orig_freq=orig_sr,
        new_freq=targ_sr,
    )
    wav = wav.squeeze(0).numpy()
    wav = wav / max(abs(wav))

    sf.write(
        targ_path,
        wav,
        targ_sr,
        subtype="PCM_24",
    )


def create_mushra_config(test_type, metadata):
    assert test_type in ["MUSHRANAT", "MUSHRAACT"]
    config = {
        "TestName": {
            "MUSHRANAT": "Speech Naturalness Evaluation (MUSHRA)",
            "MUSHRAACT": "Speech Accentedness Evaluation (MUSHRA)",
        }[test_type],
        "RateScalePng": {
            "MUSHRANAT": "img/scale_natural.png",
            "MUSHRAACT": "img/scale_accent.png",
        }[test_type],
        "RateScaleBgPng": "img/scale_abs_background.png",
        "RateMinValue": 0,
        "RateMaxValue": 100,
        "RateDefaultValue": 0,
        "ShowFileIDs": False,
        "ShowResults": False,
        "LoopByDefault": False,
        "EnableABLoop": True,
        "EnableOnlineSubmission": False,
        "BeaqleServiceURL": "/web_service/beaqleJS_Service.php",
        "SupervisorContact": "qibingbai@link.cuhk.edu.cn",
        "RandomizeTestOrder": True,
        "MaxTestsPerRun": SAMPLES_PER_TEST[test_type],
        "RequireMaxRating": False,
        "AudioRoot": "",
        "Testsets": metadata,
    }

    # Add hidden
    for item in config["Testsets"]:
        files_dict = item["Files"]
        systems = list(files_dict.keys())
        hidden_system = np.random.choice(systems)
        hidden_path = files_dict[hidden_system]
        files_dict[f"{hidden_system}---2"] = hidden_path

    return config


def create_bws_config(test_type, metadata):
    assert test_type in ["BWSSIM"]
    config = {
        "TestName": {
            "BWSSIM": "Speaker Similarity Evaluation (BWS)",
        }[test_type],
        "LoopByDefault": False,
        "ShowFileIDs": False,
        "ShowResults": False,
        "EnableABLoop": True,
        "EnableOnlineSubmission": False,
        "BeaqleServiceURL": "/web_service/beaqleJS_Service.php",
        "MaxTestsPerRun": SAMPLES_PER_TEST[test_type],
        "RandomizeTestOrder": True,
        "SupervisorContact": "qibingbai@link.cuhk.edu.cn",
        "AudioRoot": "",
        "Testsets": metadata,
    }

    # Add tips
    low_text = {
        "BWSSIM": "the 'LEAST' similar'",
    }[test_type]
    high_text = {
        "BWSSIM": "the 'MOST' similar'",
    }[test_type]
    for item in config["Testsets"]:
        item["LowText"] = low_text
        item["HighText"] = high_text
        # Specify the number of systems for each test, by default all the systems except the reference
        item["Length"] = len(
            [system for system in item["Files"] if system != "Reference"]
        )

        # Rename the systems to BWS labels
        tmp_files_dict = copy.deepcopy(item["Files"])
        files_dict = {}
        for system in tmp_files_dict.keys():
            system_info = SYSTEM_INFO_DICT[system]
            files_dict[system_info["BWS_label"]] = tmp_files_dict[system]
        item["Files"] = files_dict

    return config


def write_js_config(config, output_path):
    """Write config dict as JavaScript variable to file"""
    js_content = f"var TestConfig = {json.dumps(config, indent=2)};"
    with open(output_path, "w") as f:
        f.write(js_content)


if __name__ == "__main__":
    system2info = {item["system"]: item for item in SYSTEM_INFO}

    # Process naturalness samples
    naturalness_metadata = load_metadata(NATURALNESS_SAMPLE_LIST)
    naturalness_system_info = {
        item["system"]: item
        for item in SYSTEM_INFO
        if item["system"] in NATURALNESS_SYSTEMS
    }
    naturalness_metadata = fetch_samples(
        naturalness_metadata,
        OUTPUT_DIR,
        naturalness_system_info,
        tag="MUSHRANAT",
    )
    write_js_config(
        create_mushra_config("MUSHRANAT", naturalness_metadata),
        os.path.join(OUTPUT_DIR, "MUSHRANAT.js"),
    )

    # Process accentedness samples
    accentedness_metadata = load_metadata(ACCENTEDNESS_SAMPLE_LIST)
    accentedness_system_info = {
        item["system"]: item
        for item in SYSTEM_INFO
        if item["system"] in ACCENTEDNESS_SYSTEMS
    }
    accentedness_metadata = fetch_samples(
        accentedness_metadata,
        OUTPUT_DIR,
        accentedness_system_info,
        tag="MUSHRAACT",
    )
    write_js_config(
        create_mushra_config("MUSHRAACT", accentedness_metadata),
        os.path.join(OUTPUT_DIR, "MUSHRAACT.js"),
    )

    # Process speaker similarity samples
    spksim_metadata = load_metadata(SPKSIM_SAMPLE_LIST)
    spksim_system_info = {
        item["system"]: item for item in SYSTEM_INFO if item["system"] in SPKSIM_SYSTEMS
    }
    spksim_metadata = fetch_samples(
        spksim_metadata,
        OUTPUT_DIR,
        spksim_system_info,
        tag="BWSSIM",
    )
    write_js_config(
        create_bws_config("BWSSIM", spksim_metadata),
        os.path.join(OUTPUT_DIR, "BWSSIM.js"),
    )
