# Co-evolution Transformer
## About The Project
The implementation of the paper "Co-evolution Transformer for Protein Contact Prediction".

## Getting Started
### Prerequisites
Install [PyTorch 1.8+](https://pytorch.org/), [python3.7+](https://www.python.org/downloads/)

### Installation

1. Clone the repo
```sh
git clone xxx
```

2. Install python packages
```sh
pip install -r requirements.txt
```

## Usage
1. Generate `a3m` format MSA for a given target sequence
2. Predict contacts
```sh
run_inference.sh <MSA> <output_file>
```

## Example
```sh
cd example
./run_example.sh
```

## License
Distributed under the MIT License. See `LICENSE` for more information.
