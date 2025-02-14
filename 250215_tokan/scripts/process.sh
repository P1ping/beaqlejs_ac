#!/bin/bash

stage=2

if [ $stage == 1 ]; then
    echo "Stage 1: Data Preparation"
    python scripts/select_samples.py \
        --test_file /Users/baiqibing/Work/Interspeech2025/listening_test_interspeech2025/test-uid-text \
        --output_dir /Users/baiqibing/Work/Interspeech2025/listening_test_interspeech2025/split_metadata
fi

if [ $stage == 2 ]; then
    echo "Stage 2: Fetching Samples"
    python scripts/fetch_samples.py \

fi
