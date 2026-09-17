import librosa as lb
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import scipy.signal as scs
import scipy.ndimage as ndi
from smap.config import ETA, PHASE_PATCH_SHAPE
from smap.utils import (
    KernelParams,
    KernelType,
    KernelPoint,
    FFTParams,
    CostellationParams,
    PeaksMatrix,
    Star,
    StarPair
)


class Kernel(): # must/should be dynamic
    def __init__(self, params: KernelParams) -> None:
        self.height = params.kshape[0] # must be odd (FREQ BINS)
        self.width = params.kshape[1] # must be odd (TIME BINS)
        self.hop_height = params.khop[0]
        self.hop_width = params.khop[1]
        self.center = (self.height // 2, self.width // 2)

        self.kernel = None
        self.kernel_type: None | KernelType = None
        self.threshold = 0.0

        self.__set_kernel(mode=params.ktype, threshold=params.kthreshold)

    def __set_kernel(self, mode: KernelType, threshold: float=-81.0) -> None:
        self.kernel_type = mode
        self.threshold = threshold
        match mode:
            case KernelType.RECTANGULAR:
                self.kernel = np.ones(shape=(self.height, self.width), dtype=float) # rect
                return
            case KernelType.GAUSSIAN_ASINOT:
                y, x = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing="ij")
                centery, centerx = self.center
                sigmaf = self.height / 6.0
                sigmat = self.width / 3.0
                k = np.exp(
                    -(((y - centery) ** 2) / (2 * sigmaf ** 2)
                    + ((x - centerx) ** 2) / (2 * sigmat ** 2))
                )
                self.kernel = k / np.max(k)
                return
            case KernelType.MAXIMUM_FILTER:
                return

    def apply(self, source: NDArray) -> KernelPoint | tuple[NDArray] | None:
        if self.kernel_type == KernelType.MAXIMUM_FILTER:
            local_max = ndi.maximum_filter(source, size=(self.height, self.width))
            is_peak = (source == local_max) & (source >= self.threshold)
            peak_ys, peak_xs = np.where(is_peak)
            return (peak_xs, peak_ys)

        if self.kernel is not None and source.shape == (self.height, self.width):
            centery, centerx = self.center
            filtered = source * self.kernel
            filter_max = np.max(filtered)
            if filter_max == filtered[centery, centerx] and filter_max > self.threshold:
                return KernelPoint(x=centerx, y=centery, value=filter_max)
            else:
                return None
        return None

    def get_info_string(self) -> str:
        return f"\n[KERNEL INFO]\nTYPE: {self.kernel_type.name}\nTHRESHOLD: {self.threshold}\nWIDTH: {self.width}\nHEIGTH: {self.height}\nHOP_WIDTH: {self.hop_width}\nHOP_HEIGTH: {self.hop_height}\n"

# super-token -> [<RSTART> -> root -> [target1, target2, ...] -> <END_ROOT> -> time_vec]
class CostellationMap():
    def __init__(self, sroot: Star) -> None:
        self.root: Star = sroot
        self.targets: list[Star] = []
        self.root_token = []           # VQ anchor vector: (dB, cos, sin)
        self.root_fbin: int = 0        # anchor freq bin (tokenized directly)
        self.target_tokens = []        # VQ target vectors: (ddb, dtcos, dtsin, dt)
        self.target_dfbins = []        # target delta freq bins (tokenized directly)
        self.time_to_next = 0.0

    def append_target(self, target: Star) -> None:
        self.targets.append(target)

    def build_anchor_and_targets(self) -> None:
        acos, asin = np.cos(self.root.phase), np.sin(self.root.phase)
        self.root_fbin = int(self.root.fbin)
        self.root_token = np.array([self.root.value, acos, asin])  # (dB, cos, sin) — no freq
        target_tokens = []
        target_dfbins = []
        for target in self.targets:
            df_bin = int(target.fbin - self.root.fbin)
            ddb = target.value - self.root.value
            dtphase = target.phase - self.root.phase
            dtcos, dtsin = np.cos(dtphase), np.sin(dtphase)
            dt = target.t - self.root.t
            t = np.array([ddb, dtcos, dtsin, dt])  # (ddb, dtcos, dtsin, dt) — no df
            target_tokens.append(t)
            target_dfbins.append(df_bin)
        self.target_tokens = np.array(target_tokens)
        self.target_dfbins = np.array(target_dfbins, dtype=np.int64)

    def __repr__(self) -> str:
        r = f"ROOT: {self.root}\n"
        for target in self.targets:
            r += f"TARGET: {target}\n"
        return r

class Map():
    def __init__(self, costellation: list[CostellationMap], times: NDArray, pmatrix: PeaksMatrix, source_spectrum: NDArray) -> None:
        self.costellation = costellation
        self.times = times
        self.peaks_matrix = pmatrix
        self.source_spectrum = source_spectrum

class SonarMap():
    def __init__(self, id: str, audio_path: str | None = None, sr: int = 22050, audio_vec: NDArray | None = None) -> None:
        self.id = id
        self.audio_path = audio_path
        self.sr = sr
        self.audio_vec = audio_vec if audio_vec is not None else self.open_audio()

        self.stars: list[Star] = []
        self.pairs: list[StarPair] = []
        self.costellation_map: dict[str, CostellationMap] = {}
        self.times_vec = []

        self.peaks_matrix = PeaksMatrix(f=np.empty(0), t=np.empty(0), matrix=np.empty(0))
        self.source_spectrum: NDArray = np.empty(0)

        self.anchor_stars = []

    def open_audio(self) -> NDArray:
        if self.audio_path is None:
            return None
        try:
            audio_vec, _ = lb.load(self.audio_path, sr=self.sr, mono=True)
            return audio_vec
        except Exception as e:
            print(f"[WARNING] Not valid file: {e} skipping")
            return None

    def analyze(self, kernel: Kernel, fft_params: FFTParams) -> bool:
        if self.audio_vec is None:
            print("[ERROR] NULL audio vec!")
            return False

        try:
            real_overlap = fft_params.wsize - fft_params.hsize
            f, t, spectrum = scs.stft(self.audio_vec, window=fft_params.w, nperseg=fft_params.wsize, noverlap=real_overlap, fs=self.sr)
            pws = np.abs(spectrum)
            angle = np.angle(spectrum)

            pwsdb = 20 * np.log10(pws + ETA)
            # pwsdb_norm = pwsdb - np.max(pwsdb)

            row, col = pws.shape[0], pws.shape[1]
            self.peaks_matrix.matrix = np.zeros((row, col, 3), dtype=np.float64)
            self.peaks_matrix.f = f
            self.peaks_matrix.t = t

            # target spectrum
            self.source_spectrum = np.stack([pwsdb, np.cos(angle), np.sin(angle)], axis=-1)

            if kernel.kernel_type == KernelType.MAXIMUM_FILTER:
                map_point = kernel.apply(source=pwsdb)
                for x, y in zip(map_point[0], map_point[1]):
                    phase_val = angle[y, x]
                    cpatch = self.__get_cpatch(spectrum=spectrum, x=x, y=y)
                    star = Star(f=f[y], t=t[x], value=pwsdb[y, x], phase=phase_val, fbin=y, tbin=x, phase_patch=cpatch)
                    self.stars.append(star)
                    self.peaks_matrix.matrix[y, x] = [pwsdb[y, x], np.cos(phase_val), np.sin(phase_val)]
            else:
                for c in range(0, col - kernel.width, kernel.hop_width):
                    c_end = min(col, c + kernel.width)
                    for r in range(0, row - kernel.height, kernel.hop_height):
                        r_end = min(row, r + kernel.height)
                        source = pwsdb[r:r_end, c:c_end]
                        map_point = kernel.apply(source=source)
                        if map_point is not None:
                            x, y = c + map_point.x, r + map_point.y
                            phase_val = angle[y, x]
                            cpatch = self.__get_cpatch(spectrum=spectrum, x=x, y=y)
                            star = Star(f=f[y], t=t[x], value=map_point.value, phase=phase_val, fbin=y, tbin=x, phase_patch=cpatch)
                            self.stars.append(star)
                            self.peaks_matrix.matrix[y, x] = [map_point.value, np.cos(phase_val), np.sin(phase_val)]
            return True
        except Exception as e:
            print(f"[WARNING] Something went wrong (analyze): {e}")
            return False

    def __get_cpatch(self, spectrum, x, y) -> NDArray[np.complex64]: # patch complex
        h, w = spectrum.shape
        row_start, row_end = y - PHASE_PATCH_SHAPE[0], y + PHASE_PATCH_SHAPE[0] + 1
        col_start, col_end = x - PHASE_PATCH_SHAPE[1], x + PHASE_PATCH_SHAPE[1] + 1
        if row_start < 0 or row_end > h or col_start < 0 or col_end > w:
            return None
        patch = spectrum[row_start:row_end, col_start:col_end].astype(np.complex64)
        patch = patch / (spectrum[y, x] + ETA)
        return patch

    def __generate_pairs(self, n_pairs: int, dt_max: float) -> None:
        self.stars.sort(key=lambda s: s.tbin)
        n = len(self.stars)
        assigned = [False] * n
        self.anchor_stars = []
        for i in range(n):
            if assigned[i]:
                continue
            anchor = self.stars[i]
            self.anchor_stars.append(anchor)
            assigned[i] = True
            # all not-yet-assigned candidates in the causal future window (dt > 0)
            cands: list[int] = []
            for j in range(i + 1, n):
                dt = self.stars[j].t - anchor.t
                if dt > dt_max:
                    break
                if dt > 0.0 and not assigned[j]:
                    cands.append(j)
            if not cands:
                continue
            # stride-sample up to n_pairs of them, evenly spaced across the window
            if len(cands) <= n_pairs:
                picked = cands
            else:
                picked = [cands[round(k * (len(cands) - 1) / (n_pairs - 1))] for k in range(n_pairs)]
            for j in picked:
                self.pairs.append(StarPair(a=anchor, b=self.stars[j]))
                assigned[j] = True

    def __generate_nodes(self):
        for anchor in self.anchor_stars:
            if anchor.id not in self.costellation_map:
                self.costellation_map[anchor.id] = CostellationMap(sroot=anchor)
        for pair in self.pairs:
            cid = pair.a.id
            if cid not in self.costellation_map:
                self.costellation_map[cid] = CostellationMap(sroot=pair.a)
            self.costellation_map[cid].append_target(target=pair.b)

        for cmap in self.costellation_map.values():
            cmap.build_anchor_and_targets()

    def generate_map(self, costellation_params: CostellationParams) -> Map:
        self.__generate_pairs(n_pairs=costellation_params.n_pairs, dt_max=costellation_params.dt_max)
        self.__generate_nodes()
        cmap = sorted(self.costellation_map.values(), key=lambda item: item.root.t)
        times = []
        for i in range(1, len(cmap)):
            k1 = cmap[i - 1].root.t
            k2 = cmap[i].root.t
            times.append(k2 - k1)

        self.times_vec = times
        for i, t in enumerate(times):
            cmap[i].time_to_next = t

        if cmap:
            cmap[-1].time_to_next = 0.0
        return Map(cmap, np.array(times), self.peaks_matrix, self.source_spectrum)

    def display_map(self) -> None:
        for node in self.costellation_map.values():
            x1 = node.root.t
            y1 = node.root.f
            for target in node.targets:
                x2 = target.t
                y2 = target.f
                plt.plot([x1, x2], [y1, y2], c="k", lw=0.5, marker=".")

        plt.show()
