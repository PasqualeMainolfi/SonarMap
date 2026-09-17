# SonarMap audio codec

SonarMap is an experimental audio codec built around **frequency constellations**. It extracts sparse peaks from an STFT, represents each peak with a local complex patch, and encodes those patches with a 24-stage residual vector quantizer (RVQ). Decoding places the reconstructed patches on a complex spectrogram and converts it back to audio with ISTFT.

This repository snapshot contains the codec and its command-line round trip. It does not include a trained model checkpoint, an audio dataset, or the work-in-progress Transformer generator.

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- A compatible trained PVQ checkpoint (`pvq_best_model.pth`); model weights are not distributed with this repository
- A WAV or MP3 input file

Install the Python dependencies:

```bash
uv sync
```

## Encode and reconstruct one track

```bash
uv run python smap-process.py \
  --track input.wav \
  --pvq-model /path/to/pvq_best_model.pth \
  --out reconstructed.wav \
  --bitstream-out encoded.smbs
```

The command tokenizes the input, packs and unpacks the token stream, checks that the tokens match, then reconstructs audio. `--bitstream-out` is optional. The default bitstream mode is `lzma`; use `--mode none` for uncompressed 16-bit tokens. Input is loaded as mono audio at 22,050 Hz, and the output is a WAV file.

If `--pvq-model` is omitted, the command looks for `models/vq/pvq_best_model.pth`. That file is not included, so supply your own checkpoint or place it at the default path.

To measure bitrate across several MP3 files without writing reconstructed audio:

```bash
uv run python smap-process.py \
  --dataset-root /path/to/mp3-directory \
  --n-tracks 10 \
  --duration-s 10 \
  --pvq-model /path/to/pvq_best_model.pth
```

## Codec outline

1. `smap/cmap.py` computes the STFT and groups local spectral peaks into anchor and target constellations. Each peak carries a 13 × 5 complex patch.
2. `smap/cbook.py` quantizes scalar values with deterministic bins and encodes the patches with the PVQ model in `smap/pvq.py`.
3. `smap/bitstream.py` stores the token sequence in a lossless container, with optional LZMA compression.
4. `smap/cbook.py` decodes the tokens, rasterizes the complex patches, and reconstructs audio with `scipy.signal.istft`.

`smap-process.py` reconstructs existing audio through this codec. It does not generate new audio from a language model.
