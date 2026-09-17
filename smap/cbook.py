import numpy as np
import torch
import scipy.signal as scs
from numpy.typing import NDArray
from smap.cmap import CostellationMap
from smap.utils import CToken, FFTParams, PeaksMatrix, Star, set_torch_device
from smap.config import (
    FFT_HOPSEC,
    FFT_PARAMS,
    FFT_SRHOP,
    C_START_ID,
    C_END_ID,
    DB_MIN,
    DB_STEP,
    DB_DICT_SIZE,
    DB_OFFSET,
    APHASE_OFFSET,
    DDB_MIN,
    DDB_STEP,
    DDB_DICT_SIZE,
    DDB_OFFSET,
    TPHASE_OFFSET,
    PHASE_DICT_SIZE,
    DT_STEP,
    DT_DICT_SIZE,
    DT_OFFSET,
    FBIN_OFFSET,
    FBIN_DICT_SIZE,
    DFBIN_OFFSET,
    DFBIN_SHIFT,
    DFBIN_DICT_SIZE,
    PATCH_OFFSET,
    PATCH_DICT_SIZE,
    PATCH_H,
    PATCH_W,
    PHASE_PATCH_SHAPE,
    PVQ_N_STAGES,
    TIME_DICT_SIZE,
    TIME_SHIFT_OFFSET,
)


class DataConversion():
    TWO_PI = 2.0 * np.pi
    PHASE_STEP = TWO_PI / PHASE_DICT_SIZE

    @staticmethod
    def db_to_bin(db: float) -> int:
        idx = int(round((db - DB_MIN) / DB_STEP))
        return max(0, min(DB_DICT_SIZE - 1, idx))
    @staticmethod
    def bin_to_db(idx: int) -> float:
        return DB_MIN + int(idx) * DB_STEP
    @staticmethod
    def ddb_to_bin(ddb: float) -> int:
        idx = int(round((ddb - DDB_MIN) / DDB_STEP))
        return max(0, min(DDB_DICT_SIZE - 1, idx))
    @staticmethod
    def bin_to_ddb(idx: int) -> float:
        return DDB_MIN + int(idx) * DDB_STEP
    @staticmethod
    def phase_to_bin(phase: float) -> int:
        p = float(phase) % DataConversion.TWO_PI
        return int(round(p / DataConversion.PHASE_STEP)) % PHASE_DICT_SIZE
    @staticmethod
    def bin_to_phase(idx: int) -> float:
        return (int(idx) % PHASE_DICT_SIZE) * DataConversion.PHASE_STEP
    @staticmethod
    def dt_to_bin(dt: float) -> int:
        idx = int(round(float(dt) / DT_STEP))
        return max(0, min(DT_DICT_SIZE - 1, idx))
    @staticmethod
    def bin_to_dt(idx: int) -> float:
        return int(idx) * DT_STEP

class RasterItem():
    def __init__(self) -> None:
        self.raw_anchor: NDArray | None = None       # (dB, cos, sin)
        self.anchor_fbin: int = 0
        self.anchor_patch: NDArray | None = None     # normalized complex patch (H, W)
        self.raw_targets: list = []                  # list of (ddb, dtcos, dtsin, dt)
        self.target_dfbins: list = []                # list of delta fbin (signed)
        self.target_patches: list = []               # list of normalized complex patches
        self.time_shift = 0.0
        self.anchor: Star | None = None
        self.targets: list[Star] = []

    def add_raw_anchor(self, raw_anchor: NDArray, fbin: int, patch: NDArray | None = None) -> None:
        self.raw_anchor = raw_anchor
        self.anchor_fbin = int(fbin)
        self.anchor_patch = patch

    def add_raw_target(self, raw_target: NDArray, dfbin: int, patch: NDArray | None = None) -> None:
        self.raw_targets.append(raw_target)
        self.target_dfbins.append(int(dfbin))
        self.target_patches.append(patch)

    def add_time_shift(self, time_shift: float) -> None:
        self.time_shift = time_shift

    def from_raw_to_star(self, fft_params: FFTParams) -> None:
        if self.anchor is not None:
            return
        if self.raw_anchor is not None:
            a_db = self.raw_anchor[0]
            aphase = np.atan2(self.raw_anchor[2], self.raw_anchor[1])
            fbin = self.anchor_fbin
            f = fbin * fft_params.sr / fft_params.wsize
            t = self.time_shift
            tbin = round(t * fft_params.sr / fft_params.hsize)
            self.anchor = Star(f=f, value=a_db, t=t, phase=aphase, fbin=fbin, tbin=tbin, phase_patch=self.anchor_patch)

            for rt, dfbin, patch in zip(self.raw_targets, self.target_dfbins, self.target_patches):
                ddb = rt[0]
                dtcos = rt[1]
                dtsin = rt[2]
                dt = rt[3]
                ftbin = fbin + dfbin
                ft = ftbin * fft_params.sr / fft_params.wsize
                at_db = a_db + ddb
                dtphase = np.atan2(dtsin, dtcos)
                tphase = dtphase + aphase
                tt = t + dt
                ttbin = round(tt * fft_params.sr / fft_params.hsize)
                target = Star(f=ft, value=at_db, t=tt, phase=tphase, fbin=ftbin, tbin=ttbin, phase_patch=patch)
                self.targets.append(target)


class Rasterizer():
    def __init__(self) -> None:
        self.string_token = ""
        self.items: list[RasterItem] = []
        self.temp_item: RasterItem | None = None
        self.duration = 0.0
        self.pending_time = 0.0

    def init_item(self) -> None:
        self.temp_item = RasterItem()
        self.temp_item.add_time_shift(self.pending_time)

    def add_raw_anchor_to_item(self, raw_anchor: NDArray, fbin: int, patch: NDArray | None = None) -> None:
        if self.temp_item is not None:
            self.temp_item.add_raw_anchor(raw_anchor=raw_anchor, fbin=fbin, patch=patch)

    def add_raw_target_to_item(self, raw_target: NDArray, dfbin: int, patch: NDArray | None = None) -> None:
        if self.temp_item is not None:
            self.temp_item.add_raw_target(raw_target=raw_target, dfbin=dfbin, patch=patch)

    def add_time_shift_to_item(self, time_shift: float) -> None:
        self.duration += time_shift
        self.pending_time = self.duration

    def finalize_duration(self) -> None:
        max_time = 0.0
        for item in self.items:
            atime = item.time_shift
            if item.raw_targets:
                times = [atime + t[-1] for t in item.raw_targets]
                max_time = max(max_time, max(times))
            else:
                max_time = max(max_time, atime)
        self.duration = float(max_time)
        self.temp_item = None

    def add_string_token(self, string_token: str) -> None:
        self.string_token = string_token

    def add_item(self) -> None:
        if self.temp_item is not None:
            self.items.append(self.temp_item)
        self.temp_item = None

    def rasterize(self, fft_params: FFTParams) -> PeaksMatrix:
        n = int(self.duration * fft_params.sr)
        row = fft_params.wsize // 2 + 1
        col = int(np.floor(n / fft_params.hsize)) + 1
        rast_matrix = np.zeros(shape=(row, col, 3), dtype=np.float64)

        for item in self.items:
            item.from_raw_to_star(fft_params=fft_params)
            anchor = item.anchor
            if anchor is not None:
                fbin = min(anchor.fbin, row - 1)
                tbin = min(anchor.tbin, col - 1)
                cur = rast_matrix[fbin, tbin, 0]
                if cur == 0 or anchor.value > cur:
                    rast_matrix[fbin, tbin, 0] = anchor.value
                    rast_matrix[fbin, tbin, 1] = np.cos(anchor.phase)
                    rast_matrix[fbin, tbin, 2] = np.sin(anchor.phase)

                for target in item.targets:
                    fbin = min(target.fbin, row - 1)
                    tbin = min(target.tbin, col - 1)
                    cur = rast_matrix[fbin, tbin, 0]
                    if cur == 0 or target.value > cur:
                        rast_matrix[fbin, tbin, 0] = target.value
                        rast_matrix[fbin, tbin, 1] = np.cos(target.phase)
                        rast_matrix[fbin, tbin, 2] = np.sin(target.phase)

        f = np.fft.rfftfreq(fft_params.wsize, d=1.0 / fft_params.sr)
        t = np.arange(col) * (fft_params.hsize / fft_params.sr)
        return PeaksMatrix(f, t, rast_matrix)

    def __denormalize_cpatch(self, fft_shape: tuple[int, int], fbin: int, tbin: int, db: float, phase: float, patch, canvas):
        rf = PHASE_PATCH_SHAPE[0]
        rt = PHASE_PATCH_SHAPE[1]

        if patch is None:
            return None

        row, col = fft_shape
        y0, y1 = fbin - rf, fbin + rf + 1
        x0, x1 = tbin - rt, tbin + rt + 1

        py0, py1 = max(0, -y0), patch.shape[0] - max(0, y1 - row)
        px0, px1 = max(0, -x0), patch.shape[1] - max(0, x1 - col)
        cy0, cy1 = max(0, y0), min(row, y1)
        cx0, cx1 = max(0, x0), min(col, x1)

        if cy1 <= cy0 or cx1 <= cx0 or py1 <= py0 or px1 <= px0:
            return

        peak_complex = np.complex64((10.0 ** (db / 20.0)) * np.exp(1j * phase))
        # F1: clamp peak amplitude to 1.0 (out-of-distribution generated codes can exceed 0 dB)
        peak_mag = abs(peak_complex)
        if peak_mag > 1.0:
            peak_complex = np.complex64(peak_complex / peak_mag)
        cropped = patch[py0:py1, px0:px1].astype(np.complex64)
        # F2: clamp patch |z| max to 1.0 (degenerate codes can produce |z| > 1)
        cpatch_max = float(np.abs(cropped).max())
        if cpatch_max > 1.0:
            cropped = cropped / cpatch_max
        denorm = cropped * peak_complex
        region = canvas[cy0:cy1, cx0:cx1]
        mask = np.abs(denorm) > np.abs(region)
        region = np.where(mask, denorm, region)
        return ((cy0, cy1, cx0, cx1), region)

    def rasterize_complex(self, fft_params: FFTParams) -> NDArray[np.complex64]:
        """Complex STFT canvas built by denormalizing patches with peak complex and
        pasting at (fbin, tbin) with max-|z| overlap."""

        n = int(self.duration * fft_params.sr)
        row = fft_params.wsize // 2 + 1
        col = int(np.floor(n / fft_params.hsize)) + 1
        canvas = np.zeros((row, col), dtype=np.complex64)

        for item in self.items:
            item.from_raw_to_star(fft_params=fft_params)
            if item.anchor is not None:
                a = item.anchor
                denorm = self.__denormalize_cpatch((row, col), a.fbin, a.tbin, a.value, a.phase, a.phase_patch, canvas)
                if denorm is not None:
                    coord, region = denorm
                    canvas[coord[0]:coord[1], coord[2]:coord[3]] = region
                for s in item.targets:
                    denorm = self.__denormalize_cpatch((row, col), s.fbin, s.tbin, s.value, s.phase, s.phase_patch, canvas)
                    if denorm is not None:
                        coord, region = denorm
                        canvas[coord[0]:coord[1], coord[2]:coord[3]] = region
        return canvas


class CoEncodec():
    """Deterministic bin tokenizer. Every scalar quantity (dB, phase, dt, ddb) is
    uniformly quantized — no learned scalar VQ. Only complex patches go through a
    learned residual VQ (`PatchVQFlat`)."""

    def __init__(self, patch_vq_model_path: str | None = None, device: torch.device | None = None) -> None:
        self.device = set_torch_device() if device is None else device
        self.patch_vq = None
        if patch_vq_model_path is not None:
            from smap.pvq import PatchVQ
            self.patch_vq = PatchVQ()
            self.patch_vq.net_init()
            self.patch_vq.load_model(model_path=patch_vq_model_path)
            assert self.patch_vq.model is not None
            self.patch_vq.model.to(self.device)
            self.patch_vq.model.eval()

    def _encode_patches(self, patches: list) -> NDArray:
        n = len(patches)
        ids = np.zeros((n, PVQ_N_STAGES), dtype=np.int64)
        if self.patch_vq is None or n == 0:
            return ids
        valid_idx = [i for i, p in enumerate(patches) if p is not None and p.shape == (PATCH_H, PATCH_W)]
        if not valid_idx:
            return ids
        stacked = np.stack([np.stack([patches[i].real, patches[i].imag], axis=0) for i in valid_idx], axis=0).astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(stacked).to(self.device)
            assert self.patch_vq.model is not None
            tok = self.patch_vq.model.encode(x)
        tok = tok.detach().cpu().numpy().astype(np.int64)
        for j, idx in enumerate(valid_idx):
            ids[idx] = tok[j]
        return ids

    def _decode_patches(self, ids: NDArray) -> NDArray:
        if ids.ndim == 1:
            ids = ids.reshape(-1, PVQ_N_STAGES) if ids.size else ids.reshape(0, PVQ_N_STAGES)
        n = ids.shape[0]
        if n == 0:
            return np.zeros((0, PATCH_H, PATCH_W), dtype=np.complex64)
        if self.patch_vq is None:
            out = np.zeros((n, PATCH_H, PATCH_W), dtype=np.complex64)
            out[:, PHASE_PATCH_SHAPE[0], PHASE_PATCH_SHAPE[1]] = 1.0
            return out
        with torch.no_grad():
            t_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
            assert self.patch_vq.model is not None
            decoded = self.patch_vq.model.decode(t_ids).detach().cpu().numpy()
        decoded = decoded[:, :, :PATCH_H, :PATCH_W]
        return (decoded[:, 0] + 1j * decoded[:, 1]).astype(np.complex64)

    def tokenize(self, cmap: list[CostellationMap], sid: str) -> CToken:
        token_text = [f"<SONAR_MAP id=\"{sid}\" format=\"encoded\">\n"]
        token_tensor: list[int] = []
        total_costellations = 0

        all_anchor_patches = []
        all_target_patches = []
        for r in cmap:
            all_anchor_patches.append(r.root.phase_patch)
            for t in r.targets:
                all_target_patches.append(t.phase_patch)
        anchor_patch_ids = self._encode_patches(all_anchor_patches)
        target_patch_ids = self._encode_patches(all_target_patches)

        max_fbin = FBIN_DICT_SIZE - 1
        max_dfbin_id = DFBIN_DICT_SIZE - 1
        max_patch_id = PATCH_DICT_SIZE - 1

        # Emit initial TIME_SHIFT so first anchor's absolute time offset is preserved.
        # Without this, any leading silence (frames without peaks) collapses and the
        # reconstructed audio starts at t=0 instead of at first_anchor.t.
        if cmap:
            init_time_idx = round(float(cmap[0].root.t) * FFT_SRHOP)
            init_time_idx = max(0, min(init_time_idx, TIME_DICT_SIZE - 1))
            if init_time_idx > 0:
                token_tensor.append(init_time_idx + TIME_SHIFT_OFFSET)
                token_text.append(f"  <TIME_SHIFT index={init_time_idx + TIME_SHIFT_OFFSET}>  <!-- initial offset -->\n")

        target_global_idx = 0
        for i, r in enumerate(cmap):
            token_text.append("  <COSTELLATION>\n")
            token_tensor.append(C_START_ID)

            ar = r.root
            db_bin = DataConversion.db_to_bin(float(ar.value))
            aphase_bin = DataConversion.phase_to_bin(float(ar.phase))
            fbin_clamped = max(0, min(max_fbin, int(ar.fbin)))
            a_pids = [max(0, min(max_patch_id, int(anchor_patch_ids[i, s]))) for s in range(PVQ_N_STAGES)]

            token_tensor.append(db_bin + DB_OFFSET)
            token_tensor.append(aphase_bin + APHASE_OFFSET)
            token_tensor.append(fbin_clamped + FBIN_OFFSET)
            a_btokens = [p + PATCH_OFFSET for p in a_pids]
            token_tensor.extend(a_btokens)
            patch_str = " ".join(str(x) for x in a_btokens)
            token_text.append(
                f"    <ANCHOR db_bin={db_bin} phase_bin={aphase_bin} fbin={fbin_clamped} patch=[{patch_str}]></ANCHOR>\n"
            )

            for t in r.targets:
                ddb = float(t.value - ar.value)
                dtphase = float(t.phase - ar.phase)
                dt = float(t.t - ar.t)
                df_bin = int(t.fbin - ar.fbin)

                ddb_bin = DataConversion.ddb_to_bin(ddb)
                tphase_bin = DataConversion.phase_to_bin(dtphase)
                dt_bin = DataConversion.dt_to_bin(dt)
                dfbin_id = max(0, min(max_dfbin_id, df_bin + DFBIN_SHIFT))

                t_pids = [max(0, min(max_patch_id, int(target_patch_ids[target_global_idx, s]))) for s in range(PVQ_N_STAGES)]
                t_btokens = [p + PATCH_OFFSET for p in t_pids]

                token_tensor.append(ddb_bin + DDB_OFFSET)
                token_tensor.append(tphase_bin + TPHASE_OFFSET)
                token_tensor.append(dt_bin + DT_OFFSET)
                token_tensor.append(dfbin_id + DFBIN_OFFSET)
                token_tensor.extend(t_btokens)
                patch_str = " ".join(str(x) for x in t_btokens)
                token_text.append(
                    f"    <TARGET ddb_bin={ddb_bin} phase_bin={tphase_bin} dt_bin={dt_bin} dfbin={dfbin_id} patch=[{patch_str}]></TARGET>\n"
                )
                target_global_idx += 1

            time_shift_index = round(r.time_to_next * FFT_SRHOP)
            time_shift_index = min(time_shift_index, TIME_DICT_SIZE - 1)
            btshift = time_shift_index + TIME_SHIFT_OFFSET
            token_tensor.append(C_END_ID)
            token_tensor.append(btshift)
            token_text.append(f"  </COSTELLATION>\n  <TIME_SHIFT index={btshift}>\n")
            total_costellations += 1

        token_text.append("</SONAR_MAP>\n")
        ctoken = CToken(string_token="".join(token_text), tensor_token=np.array(token_tensor, dtype=np.int64))
        print(f"[INFO] Sonar Map [{sid}] contains: {total_costellations} costellations")
        return ctoken

    def __patch_anchor(self, cur_kind: str, cur_item: NDArray, anchor_rows_length: int) -> NDArray | None:
        if cur_kind == "anchor" and cur_item is not None:
            peak = cur_item["anchor"]
            if peak is not None and len(peak["patch_ids"]) == PVQ_N_STAGES:
                peak["patch_row"] = anchor_rows_length
                return peak["patch_ids"]
            else:
                peak["patch_row"] = -1
        return None

    def __patch_target(self, cur_kind: str, cur_item: NDArray, cur_target, target_rows_length) -> dict | None:
        if cur_kind == "target" and cur_target is not None:
            if len(cur_target["patch_ids"]) == PVQ_N_STAGES:
                cur_target["patch_row"] = target_rows_length

            else:
                cur_target["patch_row"] = -1
            return cur_target
        return None

    def detokenize(self, token: NDArray, sid: str) -> Rasterizer:
        """Parse token stream → Rasterizer.

        Two-pass for speed: first pass collects peak records and all patch-id
        rows, second pass batch-decodes all patches in one PVQ forward and
        emits items into the Rasterizer.
        """
        rast = Rasterizer()
        string_token = [f"<SONAR_MAP id=\"{sid}\" format=\"decoded\">\n"]

        # Records per costellation: each is a dict with anchor peak, list of targets, time_shift.
        items: list[dict] = []
        anchor_patch_rows: list[list[int]] = []
        target_patch_rows: list[list[int]] = []

        cur_item: dict | None = None
        cur_target: dict | None = None
        cur_kind: str | None = None  # "anchor" or "target" — which peak is open for patch ids
        initial_shift: float = 0.0  # TIME_SHIFT seen before first C_START (absolute offset to first item)

        def _new_anchor(db: float) -> dict:
            return {"db": db, "phase": None, "fbin": None, "patch_ids": [], "patch_row": None}

        def _new_target(ddb: float) -> dict:
            return {"ddb": ddb, "dphase": None, "dt": None, "dfbin": None, "patch_ids": [], "patch_row": None}

        for value in token:
            v = int(value)
            if v == C_START_ID:
                cur_target_ = self.__patch_target(cur_kind=cur_kind, cur_item=cur_item, cur_target=cur_target, target_rows_length=len(target_patch_rows))
                if cur_target_ is not None:
                    cur_target = cur_target_
                    target_patch_rows.append(cur_target["patch_ids"])
                    cur_target = None
                    cur_kind = None
                peak = self.__patch_anchor(cur_kind=cur_kind, cur_item=cur_item, anchor_rows_length=len(anchor_patch_rows))
                if peak is not None:
                    anchor_patch_rows.append(peak)
                    cur_kind = None
                cur_item = {"anchor": None, "targets": [], "time_shift": 0.0}
                items.append(cur_item)
            elif v == C_END_ID:
                cur_target_ = self.__patch_target(cur_kind=cur_kind, cur_item=cur_item, cur_target=cur_target, target_rows_length=len(target_patch_rows))
                if cur_target_ is not None:
                    cur_target = cur_target_
                    target_patch_rows.append(cur_target["patch_ids"])
                    cur_target = None
                    cur_kind = None
                peak = self.__patch_anchor(cur_kind=cur_kind, cur_item=cur_item, anchor_rows_length=len(anchor_patch_rows))
                if peak is not None:
                    anchor_patch_rows.append(peak)
                    cur_kind = None
            elif DB_OFFSET <= v < APHASE_OFFSET:
                cur_target_ = self.__patch_target(cur_kind=cur_kind, cur_item=cur_item, cur_target=cur_target, target_rows_length=len(target_patch_rows))
                if cur_target_ is not None:
                    cur_target = cur_target_
                    target_patch_rows.append(cur_target["patch_ids"])
                    cur_target = None
                    cur_kind = None
                peak = self.__patch_anchor(cur_kind=cur_kind, cur_item=cur_item, anchor_rows_length=len(anchor_patch_rows))
                if peak is not None:
                    anchor_patch_rows.append(peak)
                    cur_kind = None

                if cur_item is not None:
                    cur_item["anchor"] = _new_anchor(DataConversion.bin_to_db(v - DB_OFFSET))

                    cur_kind = "anchor"
            elif APHASE_OFFSET <= v < FBIN_OFFSET:
                if cur_item is not None and cur_item.get("anchor") is not None:
                    cur_item["anchor"]["phase"] = DataConversion.bin_to_phase(v - APHASE_OFFSET)
            elif FBIN_OFFSET <= v < DDB_OFFSET:
                if cur_item is not None and cur_item.get("anchor") is not None:
                    cur_item["anchor"]["fbin"] = int(v - FBIN_OFFSET)
            elif DDB_OFFSET <= v < TPHASE_OFFSET:
                cur_target_ = self.__patch_target(cur_kind=cur_kind, cur_item=cur_item, cur_target=cur_target, target_rows_length=len(target_patch_rows))
                if cur_target_ is not None:
                    cur_target = cur_target_
                    target_patch_rows.append(cur_target["patch_ids"])
                    cur_target = None
                    cur_kind = None
                peak = self.__patch_anchor(cur_kind=cur_kind, cur_item=cur_item, anchor_rows_length=len(anchor_patch_rows))
                if peak is not None:
                    anchor_patch_rows.append(peak)
                    cur_kind = None

                if cur_item is not None:
                    cur_target = _new_target(DataConversion.bin_to_ddb(v - DDB_OFFSET))

                    cur_item["targets"].append(cur_target)
                    cur_kind = "target"
            elif TPHASE_OFFSET <= v < DT_OFFSET:
                if cur_target is not None:
                    cur_target["dphase"] = DataConversion.bin_to_phase(v - TPHASE_OFFSET)
            elif DT_OFFSET <= v < DFBIN_OFFSET:
                if cur_target is not None:
                    cur_target["dt"] = DataConversion.bin_to_dt(v - DT_OFFSET)
            elif DFBIN_OFFSET <= v < PATCH_OFFSET:
                if cur_target is not None:
                    cur_target["dfbin"] = (v - DFBIN_OFFSET) - DFBIN_SHIFT
            elif PATCH_OFFSET <= v < TIME_SHIFT_OFFSET:
                pid = v - PATCH_OFFSET
                if cur_kind == "target" and cur_target is not None:
                    cur_target["patch_ids"].append(pid)
                elif cur_kind == "anchor" and cur_item is not None and cur_item.get("anchor") is not None:
                    cur_item["anchor"]["patch_ids"].append(pid)
            elif v >= TIME_SHIFT_OFFSET:
                cur_target_ = self.__patch_target(cur_kind=cur_kind, cur_item=cur_item, cur_target=cur_target, target_rows_length=len(target_patch_rows))
                if cur_target_ is not None:
                    cur_target = cur_target_
                    target_patch_rows.append(cur_target["patch_ids"])
                    cur_target = None
                    cur_kind = None
                peak = self.__patch_anchor(cur_kind=cur_kind, cur_item=cur_item, anchor_rows_length=len(anchor_patch_rows))
                if peak is not None:
                    anchor_patch_rows.append(peak)
                    cur_kind = None
                if cur_item is not None:
                    cur_item["time_shift"] = (v - TIME_SHIFT_OFFSET) * FFT_HOPSEC
                else:
                    # Pre-first-C_START TIME_SHIFT: absolute offset for first anchor.
                    initial_shift += (v - TIME_SHIFT_OFFSET) * FFT_HOPSEC

        # Batch decode all patches in one forward.
        anchor_patches_dec = None
        target_patches_dec = None
        if anchor_patch_rows:
            anchor_patches_dec = self._decode_patches(np.asarray(anchor_patch_rows, dtype=np.int64))
        if target_patch_rows:
            target_patches_dec = self._decode_patches(np.asarray(target_patch_rows, dtype=np.int64))

        # Apply initial offset (TIME_SHIFT emitted before first C_START) so the first
        # item gets placed at its original absolute time rather than at t=0.
        if initial_shift > 0.0:
            rast.add_time_shift_to_item(time_shift=initial_shift)

        # Emit into Rasterizer in order.
        for item in items:
            string_token.append("  <COSTELLATION>\n")
            rast.init_item()
            anchor = item.get("anchor")
            if anchor is not None and anchor["phase"] is not None and anchor["fbin"] is not None:
                patch = anchor_patches_dec[anchor["patch_row"]] if anchor["patch_row"] >= 0 and anchor_patches_dec is not None else None
                raw_anchor = np.array([anchor["db"], np.cos(anchor["phase"]), np.sin(anchor["phase"])], dtype=np.float32)
                rast.add_raw_anchor_to_item(raw_anchor=raw_anchor, fbin=int(anchor["fbin"]), patch=patch)

                string_token.append(
                    f"    <ANCHOR db={anchor['db']:.2f} phase={anchor['phase']:.4f} fbin={anchor['fbin']}></ANCHOR>\n"
                )

            for tgt in item["targets"]:
                if tgt["dphase"] is None or tgt["dt"] is None or tgt["dfbin"] is None:
                    continue
                patch = target_patches_dec[tgt["patch_row"]] if tgt["patch_row"] >= 0 and target_patches_dec is not None else None
                raw_target = np.array([tgt["ddb"], np.cos(tgt["dphase"]), np.sin(tgt["dphase"]), tgt["dt"]], dtype=np.float32)
                rast.add_raw_target_to_item(raw_target=raw_target, dfbin=int(tgt["dfbin"]), patch=patch)

                string_token.append(
                    f"    <TARGET ddb={tgt['ddb']:.2f} dphase={tgt['dphase']:.4f} dt={tgt['dt']:.5f} dfbin={tgt['dfbin']}></TARGET>\n"
                )

            string_token.append("  </COSTELLATION>\n")
            rast.add_item()
            rast.add_time_shift_to_item(time_shift=item["time_shift"])
            string_token.append(f"  <TIME_SHIFT sec={item['time_shift']:.5f}>\n")

        rast.finalize_duration()
        string_token.append("</SONAR_MAP>\n")
        rast.add_string_token(string_token="".join(string_token))
        return rast


class AudioBuilder():
    def __init__(self, peaks_matrix: PeaksMatrix | None = None, canvas: NDArray | None = None) -> None:
        self.pmatrix = peaks_matrix
        self.canvas = canvas

    def from_canvas_to_audio(self) -> NDArray[np.float64]:
        """Direct ISTFT on complex canvas produced by Rasterizer.rasterize_complex."""
        if self.canvas is None:
            raise ValueError("canvas is None — pass canvas=... to AudioBuilder")
        real_overlap = FFT_PARAMS.wsize - FFT_PARAMS.hsize
        _, audio = scs.istft(self.canvas, window=FFT_PARAMS.w, nperseg=FFT_PARAMS.wsize, noverlap=real_overlap, fs=FFT_PARAMS.sr)
        return audio
