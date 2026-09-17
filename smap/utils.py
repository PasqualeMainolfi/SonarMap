from enum  import Enum
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import torch


class StarType(Enum):
    ANCHOR = 0
    TARGET = 1

@dataclass
class KernelPoint:
    x: int
    y: int
    value: float

class KernelType(Enum):
    RECTANGULAR     = 0
    GAUSSIAN_ASINOT = 1
    MAXIMUM_FILTER      = 2

@dataclass
class KernelParams():
    ktype: KernelType
    kshape: tuple[int, int]
    khop: tuple[int, int]
    kthreshold: float

@dataclass
class CostellationParams():
    n_pairs: int
    dt_max: float

    def get_info_string(self) -> str:
        return f"\n[COSTELLATION INFO]\nN_PAIRS: {self.n_pairs}\nDT_MAX: {self.dt_max}\n"

@dataclass
class FFTParams():
    wsize: int
    hsize: int
    w: str
    sr: int

    def get_info_string(self) -> str:
        return f"\n[FFT INFO]\nWIN_SIZE: {self.wsize}\nHOP_SIZE: {self.hsize}\nWINDOW: {self.w}\nSR: {self.sr}\n"

@dataclass
class PeaksMatrix():
    f: NDArray
    t: NDArray
    matrix: NDArray

@dataclass
class Sequence():
    peaks_matrix: PeaksMatrix
    string_token: str
    canvas: NDArray | None = None

@dataclass
class Star():
    f: float
    fbin: int
    t: float
    tbin: int
    value: float
    phase: float
    phase_patch: NDArray[np.complex64] | None = None

    @property
    def id(self) -> str:
        return f"{self.tbin}|{self.fbin}"

    def __repr__(self) -> str:
        return f"f = {self.f}, t = {self.t}, v = {self.value}, bin = ({self.tbin}, {self.fbin})"

@dataclass
class StarPair():
    a: Star
    b: Star

@dataclass
class CToken():
    string_token: str
    tensor_token: NDArray

def set_torch_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    print(f"[INFO] Torch device: {device}")
    return device
