import argparse
import pathlib
import time

import numpy as np
import soundfile as sf

from smap.bitstream import MODES, kbps, pack, unpack
from smap.cbook import AudioBuilder, CoEncodec, Rasterizer
from smap.cmap import Kernel, SonarMap
from smap.utils import CToken
from smap.config import (
    COSTELLATION_PARAMS,
    FFT_PARAMS,
    KERNEL_PARAMS,
    PVQ_MODEL_PATH,
)


def _discover(root: str, n: int) -> list[str]:
    return [str(p) for p in sorted(pathlib.Path(root).rglob("*.mp3")) if not p.name.startswith("._")][:n]

def _load(path: str, sr: int) -> np.ndarray:
    import librosa
    a, _ = librosa.load(path, sr=sr, mono=True)
    return a.astype(np.float32)

def _tokenize(path: str, dur_s: float, kernel: Kernel, enc: CoEncodec, sid: str) -> tuple[np.ndarray, float]:
    audio = _load(path, FFT_PARAMS.sr)[:int(dur_s * FFT_PARAMS.sr)] if dur_s > 0 else _load(path, FFT_PARAMS.sr)
    smap = SonarMap(id=sid, sr=FFT_PARAMS.sr, audio_vec=audio)
    smap.analyze(kernel=kernel, fft_params=FFT_PARAMS)
    cmap = smap.generate_map(costellation_params=COSTELLATION_PARAMS).costellation
    ct: CToken = enc.tokenize(cmap=cmap, sid=sid)
    tt = ct.tensor_token
    tok = tt.cpu().numpy().astype(np.int64) if hasattr(tt, "cpu") else np.asarray(tt, dtype=np.int64)
    return tok, len(audio) / FFT_PARAMS.sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="lzma")
    ap.add_argument("--track", help="input WAV or MP3 file for a single-track round trip")
    ap.add_argument("--dataset-root", help="directory of MP3 files for a pooled bitrate run")
    ap.add_argument("--pvq-model", default=PVQ_MODEL_PATH,
                    help=f"PVQ checkpoint (default: {PVQ_MODEL_PATH})")
    ap.add_argument("--out", default="test_latest.wav")
    ap.add_argument("--bitstream-out", default=None)
    ap.add_argument("--n-tracks", type=int, default=1,
                    help="if >1, pool N tracks from --dataset-root and report kbps (no audio write)")
    ap.add_argument("--duration-s", type=float, default=10.0,
                    help="per-track duration when --n-tracks>1")
    args = ap.parse_args()

    if args.n_tracks < 1:
        ap.error("--n-tracks must be at least 1")
    if args.n_tracks > 1:
        if not args.dataset_root:
            ap.error("--dataset-root is required when --n-tracks is greater than 1")
        if not pathlib.Path(args.dataset_root).is_dir():
            ap.error(f"dataset directory does not exist: {args.dataset_root}")
    elif not args.track:
        ap.error("--track is required for a single-track round trip")
    elif not pathlib.Path(args.track).is_file():
        ap.error(f"input audio file does not exist: {args.track}")
    if not pathlib.Path(args.pvq_model).is_file():
        ap.error(f"PVQ checkpoint does not exist: {args.pvq_model}; pass --pvq-model")

    kernel = Kernel(params=KERNEL_PARAMS)
    encodec = CoEncodec(patch_vq_model_path=args.pvq_model)

    if args.n_tracks > 1:
        files = _discover(args.dataset_root, args.n_tracks)
        if not files:
            ap.error(f"no MP3 files found under {args.dataset_root}")
        print(f"[pooled] {len(files)} tracks × {args.duration_s:.1f}s")
        pooled: list[np.ndarray] = []
        total_dur = 0.0
        for i, p in enumerate(files):
            t0 = time.time()
            tok, d = _tokenize(p, args.duration_s, kernel, encodec, sid=f"p_{i}")
            pooled.append(tok)
            total_dur += d
            print(f"  [{i+1}/{len(files)}] n={len(tok):,}  ({time.time()-t0:.1f}s)")
        tokens = np.concatenate(pooled)
        blob = pack(tokens, mode=args.mode)
        tokens_rt = unpack(blob)
        if not np.array_equal(tokens, tokens_rt):
            raise RuntimeError("bitstream roundtrip mismatch")
        raw_kbps = tokens.shape[0] * 16.0 / total_dur / 1000.0
        print(f"\n[bitstream] mode={args.mode}  n_tok={tokens.shape[0]:,}  "
              f"bytes={len(blob):,}  kbps={kbps(blob, total_dur):.1f}  (raw_u16={raw_kbps:.1f})")
        if args.bitstream_out:
            pathlib.Path(args.bitstream_out).write_bytes(blob)
            print(f"[bitstream] wrote {args.bitstream_out}")
        return

    # Single-track path: full round-trip with audio output.
    smap = SonarMap(id="single", audio_path=args.track, sr=FFT_PARAMS.sr)
    smap.analyze(kernel=kernel, fft_params=FFT_PARAMS)
    cmap = smap.generate_map(costellation_params=COSTELLATION_PARAMS)
    ctoken: CToken = encodec.tokenize(cmap=cmap.costellation, sid="test")
    tokens = np.asarray(ctoken.tensor_token, dtype=np.int64)

    # lzma compression
    blob = pack(tokens, mode=args.mode)
    tokens_rt = unpack(blob)
    if not np.array_equal(tokens, tokens_rt):
        raise RuntimeError("bitstream roundtrip mismatch")

    dur_s = len(smap.audio_vec) / FFT_PARAMS.sr if smap.audio_vec is not None else 0.0
    raw_kbps = tokens.shape[0] * 16.0 / dur_s / 1000.0 if dur_s > 0 else 0.0
    print(f"[bitstream] mode={args.mode}  n_tok={tokens.shape[0]:,}  "
          f"bytes={len(blob):,}  kbps={kbps(blob, dur_s):.1f}  (raw_u16={raw_kbps:.1f})")

    if args.bitstream_out:
        pathlib.Path(args.bitstream_out).write_bytes(blob)
        print(f"[bitstream] wrote {args.bitstream_out}")

    detoken: Rasterizer = encodec.detokenize(tokens_rt, sid="test")
    canvas = detoken.rasterize_complex(fft_params=FFT_PARAMS)
    audio_vec = AudioBuilder(canvas=canvas).from_canvas_to_audio()
    sf.write(args.out, audio_vec, FFT_PARAMS.sr)
    print(f"[audio] wrote {args.out}")


if __name__ == "__main__":
    main()
