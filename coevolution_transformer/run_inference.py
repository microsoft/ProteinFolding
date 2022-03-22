#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import subprocess

import torch
import numpy as np

from model.model import Model


def get_a3m_feat(path):
    with open(path) as fp:
        line = fp.readline()
        if line.startswith(">"):
            line = fp.readline()
        L = len(line.strip())
    program = [
        os.path.join(os.path.dirname(__file__), "bin/a3m_to_feat"),
        "--input",
        path,
        "--max_gap",
        "7",
        "--max_keep",
        "5000",
        "--sample_ratio",
        "1.0",
    ]
    process = subprocess.run(program, capture_output=True)
    assert process.returncode == 0, "Invalid A3M file"
    x = np.copy(np.frombuffer(process.stdout, dtype=np.int8))
    x = x.reshape((-1, L, 7 * 2 + 3)).transpose((0, 2, 1))
    assert (x < 23).all(), "Internal error"
    seq = x[0][0]
    return {
        "seq": torch.tensor(seq).long()[None],
        "msa": torch.tensor(x).long()[None],
        "index": torch.arange(seq.shape[0]).long()[None],
    }


def load_model():
    model = Model().eval()
    weight_path = os.path.join(os.path.dirname(__file__), "weights.chk")
    state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    return model


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_inference.py <a3m> <output_file>")
        sys.exit(1)
    feat = get_a3m_feat(sys.argv[1])
    model = load_model()
    with torch.no_grad():
        output = model(feat)
    contact = torch.sum(output[..., :12], dim=-1)[0]
    np.savetxt(sys.argv[2], contact)


if __name__ == "__main__":
    main()
