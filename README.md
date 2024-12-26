# muzero
An unofficial reimplementation of MuZero, Gumbel MuZero in Pytorch.

## Requirements
The project is running on Python 3.10. To install dependencies, run the following command
```bash
pip install -r requirements.txt
```

## Running experiments
Each experiment can be run by calling `main.py`, choosing mode (`train` or `test`), and either with required arguments or with a predefined config file with tag `--config-path`. For instance:
```bash
python main.py train --config-path configs/train/muzero_cartpole.json
```

## Acknowledgements
The code is heavily inspired by these repos:
- [muzero-general](https://github.com/werner-duvaud/muzero-general)
- [mctx](https://github.com/google-deepmind/mctx)

## References
[1] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, David Silver et al. [Mastering Atari, Go, chess and shogi by planning with a learned model](https://doi.org/10.1038/s41586-020-03051-4). Nature 588, 604–609, 2020.
[2] Ivo Danihelka, Arthur Guez, Julian Schrittwieser, David Silver. [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO). ICLR, 2022