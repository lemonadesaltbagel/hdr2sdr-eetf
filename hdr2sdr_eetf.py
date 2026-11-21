#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hdr2sdr_eetf.py — 用 ITU-R BT.2390 EETF 把 HDR 照片转换为 SDR 照片

设计原则
--------
* 只解码 HDR 主图像素（PQ / HLG / 线性浮点），**完全不读取、不参考、不写出** gain map：
  - 不解析 MPF / XMP hdrgm / ISO 21496-1 / Apple 辅助图等任何增益图数据；
  - 不使用文件里自带的 SDR 基础图（如果主图本身就是 SDR，程序会报错退出，
    因为这类文件的 HDR 信息只存在于 gain map 中）；
  - 输出文件重新编码，只带 ICC（sRGB 或 Display P3）与可选的 EXIF，不含任何 HDR 元数据。
* 色调映射严格使用 BT.2390 EETF（PQ 域、Hermite 样条高光滚降、可选暗部 toe）。
* 先把 HDR 绝对亮度中的“参考白”（默认 203 nit，BT.2408）对齐到 SDR 的 100% 白，
  因此拐点以下的中间调与肤色 1:1 保留，只压缩高光。

支持的输入
----------
  .avif .heic .heif .hif .jxl .png(cICP) .tif/.tiff .exr
  传递函数自动从 CICP / nclx / ICC 'cicp' 标签识别，也可用 --transfer/--primaries 手动指定。

用法示例
--------
  python hdr2sdr_eetf.py in.avif                      # 输出 in_sdr.jpg（sRGB）
  python hdr2sdr_eetf.py in.heic -o out.png --bits 16 --gamut p3
  python hdr2sdr_eetf.py merge.exr --linear-white 203 --source-peak 2000
  python hdr2sdr_eetf.py in.jxl --mode luminance --knee-offset 1.0
"""
from __future__ import annotations

__version__ = "1.1.0"

import argparse
import io
import os
import struct
import sys
import zlib
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# CICP 常量（ITU-T H.273）
# ---------------------------------------------------------------------------
TC_BT709 = {1, 6, 14, 15}
TC_SRGB = 13
TC_LINEAR = 8
TC_PQ = 16
TC_HLG = 18

# xy 色度坐标：R, G, B, W
PRIMARIES = {
    1: ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), (0.3127, 0.3290)),   # BT.709 / sRGB
    9: ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), (0.3127, 0.3290)),   # BT.2020 / 2100
    12: ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), (0.3127, 0.3290)),  # Display P3 (D65)
}
PRIMARY_NAMES = {1: "BT.709", 9: "BT.2020", 12: "Display P3"}
TRANSFER_NAMES = {TC_PQ: "PQ (ST 2084)", TC_HLG: "HLG", TC_LINEAR: "Linear", TC_SRGB: "sRGB"}


# ---------------------------------------------------------------------------
# 色彩数学
# ---------------------------------------------------------------------------
def rgb_to_xyz_matrix(prim_id: int) -> np.ndarray:
    (rx, ry), (gx, gy), (bx, by), (wx, wy) = PRIMARIES[prim_id]
    xyz = lambda x, y: np.array([x / y, 1.0, (1 - x - y) / y])
    m = np.stack([xyz(rx, ry), xyz(gx, gy), xyz(bx, by)], axis=1)
    s = np.linalg.solve(m, xyz(wx, wy))
    return m * s


def conversion_matrix(src: int, dst: int) -> np.ndarray:
    if src == dst:
        return np.eye(3)
    # 三种原色同为 D65 白点，无需色适应
    return np.linalg.inv(rgb_to_xyz_matrix(dst)) @ rgb_to_xyz_matrix(src)


# PQ (SMPTE ST 2084)
_M1, _M2 = 2610 / 16384, 2523 / 4096 * 128
_C1, _C2, _C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def pq_eotf(e: np.ndarray) -> np.ndarray:
    """PQ 码值 [0,1] → 绝对亮度 nit"""
    e = np.clip(e, 0.0, 1.0)
    p = np.power(e, 1.0 / _M2)
    return 10000.0 * np.power(np.maximum(p - _C1, 0.0) / (_C2 - _C3 * p), 1.0 / _M1)


def pq_inv_eotf(lum: np.ndarray) -> np.ndarray:
    """绝对亮度 nit → PQ 码值 [0,1]"""
    y = np.power(np.clip(np.asarray(lum, dtype=np.float64) / 10000.0, 0.0, 1.0), _M1)
    return np.power((_C1 + _C2 * y) / (1.0 + _C3 * y), _M2)


# HLG (BT.2100)
_HA, _HB, _HC = 0.17883277, 0.28466892, 0.55991073


def hlg_inv_oetf(e: np.ndarray) -> np.ndarray:
    e = np.clip(e, 0.0, 1.0)
    return np.where(e <= 0.5, e * e / 3.0, (np.exp((e - _HC) / _HA) + _HB) / 12.0)


def hlg_to_display_nits(rgb_sig: np.ndarray, lw: float, luma: np.ndarray) -> np.ndarray:
    """HLG 信号 → 显示光（nit），使用 BT.2100 OOTF，峰值 lw"""
    scene = hlg_inv_oetf(rgb_sig)
    gamma = 1.2 + 0.42 * np.log10(lw / 1000.0)
    ys = np.maximum(scene @ luma, 1e-12)[..., None]
    return lw * np.power(ys, gamma - 1.0) * scene


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# BT.2390 EETF
# ---------------------------------------------------------------------------
@dataclass
class EETF:
    """
    ITU-R BT.2390 EETF（PQ 域）。
      Lw / Lb   : 源（母版）峰值 / 黑位，nit
      Lmax/Lmin : 目标峰值 / 黑位，nit
      knee_offset: KS = (1+k)*maxLum - k；BT.2390 规范值 k = 0.5（即 1.5*maxLum - 0.5）
    """
    Lw: float
    Lmax: float
    Lb: float = 0.0
    Lmin: float = 0.0
    knee_offset: float = 0.5
    pq_w: float = field(init=False)
    pq_b: float = field(init=False)
    max_lum: float = field(init=False)
    min_lum: float = field(init=False)
    ks: float = field(init=False)

    def __post_init__(self):
        self.pq_w = float(pq_inv_eotf(self.Lw))
        self.pq_b = float(pq_inv_eotf(self.Lb))
        span = self.pq_w - self.pq_b
        self.max_lum = (float(pq_inv_eotf(self.Lmax)) - self.pq_b) / span
        self.min_lum = (float(pq_inv_eotf(self.Lmin)) - self.pq_b) / span
        k = self.knee_offset
        self.ks = float(np.clip((1 + k) * self.max_lum - k, 0.0, self.max_lum))

    @property
    def knee_nits(self) -> float:
        return float(pq_eotf(self.ks * (self.pq_w - self.pq_b) + self.pq_b))

    def __call__(self, lum: np.ndarray) -> np.ndarray:
        span = self.pq_w - self.pq_b
        e1 = np.clip((pq_inv_eotf(lum) - self.pq_b) / span, 0.0, 1.0)
        ks, ml = self.ks, self.max_lum
        if ks < 1.0:
            t = (e1 - ks) / (1.0 - ks)
            t2, t3 = t * t, t * t * t
            p = ((2 * t3 - 3 * t2 + 1) * ks
                 + (t3 - 2 * t2 + t) * (1.0 - ks)
                 + (-2 * t3 + 3 * t2) * ml)
            e2 = np.where(e1 < ks, e1, p)
        else:
            e2 = e1
        if self.min_lum > 0:                     # 暗部 toe（BT.2390 黑位抬升）
            e2 = e2 + self.min_lum * np.power(np.clip(1.0 - e2, 0, 1), 4)
        e4 = e2 * span + self.pq_b
        return pq_eotf(e4)


# ---------------------------------------------------------------------------
# 解码：只取 HDR 主图像素
# ---------------------------------------------------------------------------
@dataclass
class Decoded:
    raw: np.ndarray             # HxWx3 原始码值（uint8/uint16/float），不做整体拷贝
    code_max: float             # 码值满幅（对应归一化 1.0）；浮点输入为 1.0
    transfer: int | None
    primaries: int | None
    exif: bytes | None = None
    orientation: int = 1        # 尚需由本程序应用的 EXIF 方向
    note: str = ""


def _cicp_from_icc(icc: bytes | None):
    """从 ICC v4.4 的 'cicp' 标签读取 (primaries, transfer)"""
    if not icc or len(icc) < 132:
        return None
    n = struct.unpack(">I", icc[128:132])[0]
    for i in range(min(n, 200)):
        off = 132 + 12 * i
        sig, toff, tsize = struct.unpack(">4sII", icc[off:off + 12])
        if sig == b"cicp" and toff + 12 <= len(icc):
            return icc[toff + 8], icc[toff + 9]
    return None


def _raw(a: np.ndarray, bits: int | None):
    """返回 (原始数组, 满幅码值)。整数输入保持原样，后续用查找表转换"""
    if a.dtype.kind == "f":
        return a, 1.0
    if a.dtype.itemsize > 2:                      # uint32 等：查找表太大，转成浮点
        return (a.astype(np.float32) / float(np.iinfo(a.dtype).max)), 1.0
    if bits is None:
        bits = 8 * a.dtype.itemsize
    return a, float((1 << bits) - 1)


def _rgb3(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.shape[2] == 4:            # 丢弃 alpha
        a = a[..., :3]
    elif a.shape[2] == 2:
        a = np.repeat(a[..., :1], 3, axis=2)
    return a


def _pil_side_info(path: str):
    """用 Pillow 读取 EXIF 与方向（不读取任何像素以外的 HDR 数据）"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            orient = int(ex.get(0x0112, 1)) if ex else 1
            return (ex.tobytes() if ex else None), orient
    except Exception:
        return None, 1


_PNG_PROBE = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010800000000"
    "3a7e9b550000000a49444154789c63600000000200015a8c6a6e0000000049454e44ae426082")


def decode(path: str) -> Decoded:
    ext = os.path.splitext(path)[1].lower()
    data = open(path, "rb").read()

    if ext in (".heic", ".heif", ".hif"):
        import pillow_heif
        hf = pillow_heif.open_heif(path, convert_hdr_to_8bit=False)  # 只取主图；辅助图/增益图不读
        arr = np.asarray(hf)
        bd = int(hf.info.get("bit_depth") or 0)
        if not bd and arr.dtype == np.uint16:    # 无位深信息：按所有像素共同的低位零位数推断
            ored = int(np.bitwise_or.reduce(arr[::7, ::7].ravel())) or 1
            bd = 16 - min(8, (ored & -ored).bit_length() - 1)
        bd = bd or 8
        if arr.dtype == np.uint16:   # pillow-heif 把 10/12bit 左移到 16bit 高位：满幅码值同样左移即可
            cmax = float(((1 << bd) - 1) << (16 - bd))
        else:
            cmax = 255.0
        nclx = hf.info.get("nclx_profile") or {}
        tc, pr = nclx.get("transfer_characteristics"), nclx.get("color_primaries")
        if tc is None:
            c = _cicp_from_icc(hf.info.get("icc_profile"))
            if c:
                pr, tc = c
        return Decoded(_rgb3(arr), cmax, tc, pr, hf.info.get("exif"), 1, "HEIF (libheif 已应用旋转)")

    import imagecodecs as ic
    try:
        ic.png_decode(_PNG_PROBE, _info=True)
    except TypeError:
        raise SystemExit(f"错误：需要 imagecodecs >= 2026.11.11（当前 {ic.__version__}）才能读取 CICP 色彩信息。\n"
                         "请运行：pip install -U imagecodecs（需要 Python >= 3.12）")
    if ext == ".avif":
        info = ic.avif_decode(data, _info=True)
        arr, cmax = _raw(ic.avif_decode(data, numthreads=_threads()), info.get("bitspersample"))
        tc, pr = info.get("transfer"), info.get("primaries")
        if (tc in (None, 2)) and info.get("iccprofile"):
            c = _cicp_from_icc(info["iccprofile"])
            if c:
                pr, tc = c
        exif, orient = _pil_side_info(path)   # libavif 不应用 irot/imir，这里自行处理
        return Decoded(_rgb3(arr), cmax, int(tc) if tc is not None else None,
                       int(pr) if pr is not None else None, exif, orient, "AVIF")

    if ext == ".jxl":
        info = ic.jpegxl_decode(data, _info=True)
        arr, cmax = _raw(ic.jpegxl_decode(data, numthreads=_threads()), info.get("bitspersample"))  # libjxl 已应用方向
        tc, pr = info.get("transfer"), info.get("primaries")
        if tc in (None, 2) or pr in (None, 2):
            c = _cicp_from_icc(info.get("iccprofile"))
            if c:
                pr, tc = c
        return Decoded(_rgb3(arr), cmax, int(tc) if tc is not None else None,
                       int(pr) if pr is not None else None, None, 1, "JPEG XL")

    if ext == ".png":
        info = ic.png_decode(data, _info=True)
        arr, cmax = _raw(ic.png_decode(data), info.get("bit_depth"))
        tc, pr = info.get("transfer"), info.get("primaries")
        if tc is None:
            c = _cicp_from_icc(info.get("iccprofile"))
            if c:
                pr, tc = c
        exif, orient = _pil_side_info(path)
        return Decoded(_rgb3(arr), cmax, int(tc) if tc is not None else None,
                       int(pr) if pr is not None else None, exif, orient, "PNG")

    if ext in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            arr = page.asarray()
            icc = page.tags.get(34675)
            icc = icc.value if icc else None
            orient = int(page.tags[274].value) if 274 in page.tags else 1
        tc = pr = None
        c = _cicp_from_icc(icc)
        if c:
            pr, tc = c
        if arr.dtype.kind == "f" and tc is None:
            tc = TC_LINEAR
        exif, _ = _pil_side_info(path)
        arr, cmax = _raw(arr, None)
        return Decoded(_rgb3(arr), cmax, tc, pr, exif, orient, "TIFF")

    if ext == ".exr":
        arr = ic.exr_decode(data)              # float16/float32 保持原样，分块时再转换
        return Decoded(_rgb3(arr), 1.0, TC_LINEAR, None, None, 1, "OpenEXR（线性）")

    if ext in (".jpg", ".jpeg"):
        raise SystemExit(
            "错误：JPEG（含 Ultra HDR / ISO 21496-1 增益图 JPEG）的主图是 SDR，HDR 信息只存在于 gain map 中。\n"
            "本程序按设计不读取 gain map，因此无法处理此类文件。请提供 PQ/HLG 编码的 AVIF/HEIF/JXL/PNG，或线性 EXR/TIFF。")
    raise SystemExit(f"错误：不支持的输入格式 {ext}")


def apply_orientation(a: np.ndarray, o: int) -> np.ndarray:
    if o == 2: return a[:, ::-1]
    if o == 3: return a[::-1, ::-1]
    if o == 4: return a[::-1, :]
    if o == 5: return np.transpose(a, (1, 0, 2))
    if o == 6: return np.rot90(a, -1)
    if o == 7: return np.rot90(a, 1)[::-1, :]
    if o == 8: return np.rot90(a, 1)
    return a


# ---------------------------------------------------------------------------
# ICC v4 生成（matrix/TRC，sRGB 曲线）
# ---------------------------------------------------------------------------
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                      [-0.7502, 1.7135, 0.0367],
                      [0.0389, -0.0685, 1.0296]])
_D50 = np.array([0.9642, 1.0, 0.8249])


def _s15(v: float) -> bytes:
    return struct.pack(">i", int(round(v * 65536)))


def _mluc(text: str) -> bytes:
    s = text.encode("utf-16-be")
    return b"mluc" + b"\0" * 4 + struct.pack(">II", 1, 12) + b"enUS" + struct.pack(">II", len(s), 28) + s


def build_icc(prim_id: int, name: str) -> bytes:
    m = rgb_to_xyz_matrix(prim_id)
    wxyz = m @ np.ones(3)
    lms_s, lms_d = _BRADFORD @ wxyz, _BRADFORD @ _D50
    chad = np.linalg.inv(_BRADFORD) @ np.diag(lms_d / lms_s) @ _BRADFORD
    m50 = chad @ m
    xyz = lambda v: b"XYZ " + b"\0" * 4 + b"".join(_s15(x) for x in v)
    para = (b"para" + b"\0" * 4 + struct.pack(">HH", 3, 0)
            + b"".join(_s15(v) for v in (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045)))
    tags = [
        (b"desc", _mluc(name)),
        (b"cprt", _mluc("No copyright, use freely")),
        (b"wtpt", xyz(_D50)),
        (b"chad", b"sf32" + b"\0" * 4 + b"".join(_s15(v) for v in chad.flatten())),
        (b"rXYZ", xyz(m50[:, 0])), (b"gXYZ", xyz(m50[:, 1])), (b"bXYZ", xyz(m50[:, 2])),
        (b"rTRC", para), (b"gTRC", para), (b"bTRC", para),
    ]
    table, body = b"", b""
    off = 128 + 4 + 12 * len(tags)
    shared = {}
    for sig, d in tags:
        if d in shared:
            o, ln = shared[d]
        else:
            pad = (-len(body)) % 4
            body += b"\0" * pad
            o, ln = off + len(body), len(d)
            body += d
            shared[d] = (o, ln)
        table += sig + struct.pack(">II", o, ln)
    total = 128 + 4 + len(table) + len(body)
    hdr = (struct.pack(">I", total) + b"\0" * 4 + struct.pack(">I", 0x04300000) + b"mntrRGB XYZ "
           + struct.pack(">6H", 2026, 1, 1, 0, 0, 0) + b"acsp" + b"\0" * 24
           + struct.pack(">I", 0) + b"".join(_s15(v) for v in _D50) + b"\0" * 4 + b"\0" * 16 + b"\0" * 28)
    assert len(hdr) == 128
    return hdr + struct.pack(">I", len(tags)) + table + body


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def gamut_fit(rgb: np.ndarray, luma: np.ndarray, method: str) -> np.ndarray:
    """把线性 RGB 放进 [0,1]：compress = 沿亮度轴向灰色收拢（保持亮度与色相），clip = 硬裁切"""
    if method == "clip":
        return np.clip(rgb, 0.0, 1.0)
    dt = rgb.dtype
    y = np.clip(rgb[..., 0] * luma[0] + rgb[..., 1] * luma[1] + rgb[..., 2] * luma[2], 0.0, 1.0).astype(dt)
    d = rgb - y[..., None]
    yb = np.broadcast_to(y[..., None], rgb.shape)
    t = np.full(rgb.shape, np.inf, dtype=dt)          # 每个通道允许的最大混合比例
    neg, over = rgb < 0, rgb > 1
    t[neg] = yb[neg] / -d[neg]                        # rgb<0 ⇒ d<0
    t[over] = (1 - yb[over]) / d[over]                # rgb>1 ⇒ d>0
    tm = np.clip(np.minimum(np.minimum(t[..., 0], t[..., 1]), t[..., 2]), 0.0, 1.0)
    out = d
    out *= tm[..., None]
    out += y[..., None]
    return np.clip(out, 0.0, 1.0, out=out)


# ---------------------------------------------------------------------------
# 快速管线：查找表 + float32 + 多线程分块
# ---------------------------------------------------------------------------
_F = np.float32
_STRIP = 256                       # 每块行数；抖动噪声表的周期与之相同


def _threads(n: int | None = None) -> int:
    n = n or getattr(_threads, "override", 0) or os.cpu_count() or 1
    return max(1, int(n))


def _max3(a: np.ndarray) -> np.ndarray:
    # 比 a.max(axis=-1) 快数倍（长度为 3 的末轴归约在 numpy 中很慢）
    return np.maximum(np.maximum(a[..., 0], a[..., 1]), a[..., 2])


def _min3(a: np.ndarray) -> np.ndarray:
    return np.minimum(np.minimum(a[..., 0], a[..., 1]), a[..., 2])


def _dot3(a: np.ndarray, w) -> np.ndarray:
    return a[..., 0] * _F(w[0]) + a[..., 1] * _F(w[1]) + a[..., 2] * _F(w[2])


def _pq_eotf32(e: np.ndarray) -> np.ndarray:
    e = np.clip(e.astype(_F, copy=False), 0, 1)
    p = np.power(e, _F(1 / _M2))
    return _F(10000) * np.power(np.maximum(p - _F(_C1), 0) / (_F(_C2) - _F(_C3) * p), _F(1 / _M1))


def _hlg_inv_oetf32(e: np.ndarray) -> np.ndarray:
    e = np.clip(e.astype(_F, copy=False), 0, 1)
    return np.where(e <= 0.5, e * e / _F(3), (np.exp((e - _F(_HC)) / _F(_HA)) + _F(_HB)) / _F(12)).astype(_F)


class _LogLUT:
    """一元函数 f(L) 在 log2(L) 均匀网格上的表 + 线性插值（比 np.interp 快约 10 倍）。
    网格覆盖 1e-6 … 1.3e8 nit，约 350 点/档，相对误差 < 1e-6。"""
    LO, HI, N = -20.0, 27.0, 1 << 14

    def __init__(self, fn):
        x = np.linspace(self.LO, self.HI, self.N)
        t = np.asarray(fn(np.exp2(x)), dtype=np.float64)
        self.t = t.astype(_F)
        self.d = np.append(np.diff(t), 0.0).astype(_F)
        self.k = _F((self.N - 1) / (self.HI - self.LO))

    def __call__(self, lum: np.ndarray) -> np.ndarray:
        f = np.log2(np.maximum(lum, _F(1e-30)))
        f -= _F(self.LO)
        f *= self.k
        np.clip(f, 0, self.N - 1, out=f)
        i = f.astype(np.int32)
        f -= i
        f *= self.d[i]
        f += self.t[i]
        return f


def _srgb_oetf32(x: np.ndarray) -> np.ndarray:
    hi = np.power(x, _F(1 / 2.4))
    hi *= _F(1.055)
    hi -= _F(0.055)
    lo = x <= _F(0.0031308)
    hi[lo] = x[lo] * _F(12.92)
    return hi


# 8 bit 编码表：线性值量化到 1/65535 后查表（近黑处步长 < 0.06 级，远小于 8 bit 精度）
_SRGB8 = (srgb_oetf(np.arange(65536) / 65535.0) * 255.0 + 0.5).astype(_F)


def _resolve_color(dec: Decoded, args):
    tc = args.transfer if args.transfer is not None else dec.transfer
    pr = args.primaries if args.primaries is not None else dec.primaries
    tc = {"pq": TC_PQ, "hlg": TC_HLG, "linear": TC_LINEAR}.get(tc, tc)
    pr = {"bt2020": 9, "p3": 12, "bt709": 1}.get(pr, pr)
    if pr is None or pr == 2:
        pr = 9 if tc in (TC_PQ, TC_HLG) else 1
        print(f"提示：未检测到色彩原色，假定为 {PRIMARY_NAMES[pr]}（可用 --primaries 指定）")
    if pr not in PRIMARIES:
        raise SystemExit(f"错误：不支持的色彩原色 CICP={pr}")
    if tc in TC_BT709 or tc == TC_SRGB or tc is None or tc == 2:
        raise SystemExit(
            f"错误：主图传递函数为 {TRANSFER_NAMES.get(tc, tc)}，不是 HDR。\n"
            "此文件的 HDR 信息很可能只存在于 gain map 中（如 iPhone HEIC、Ultra HDR），"
            "而本程序按设计不读取 gain map。\n若确认像素是 PQ/HLG/线性，请用 --transfer 手动指定。")
    if tc not in (TC_PQ, TC_HLG, TC_LINEAR):
        raise SystemExit(f"错误：不支持的传递函数 CICP={tc}")
    return tc, pr


def process(args) -> None:
    _threads.override = args.threads
    dec = decode(args.input)
    tc, pr = _resolve_color(dec, args)
    raw = dec.raw
    h, w = raw.shape[:2]
    out_pr = {"srgb": 1, "p3": 12}[args.gamut]
    luma_src = rgb_to_xyz_matrix(pr)[1]
    luma_out = rgb_to_xyz_matrix(out_pr)[1].astype(_F)
    M = conversion_matrix(pr, out_pr)
    Mt = None if np.allclose(M, np.eye(3)) else np.ascontiguousarray(M.T, dtype=_F)
    inv_white = 1.0 / args.sdr_white

    # ---- 原始码值 → 绝对亮度（nit）。整数输入全部走查找表，无逐像素幂运算 ----
    is_int = raw.dtype.kind in "ui"
    code_nits = code_scene = None
    if is_int:
        norm = np.clip(np.arange(1 << (8 * raw.dtype.itemsize)) / dec.code_max, 0.0, 1.0)
        if tc == TC_PQ:
            code_nits = pq_eotf(norm)
        elif tc == TC_LINEAR:
            code_nits = norm * args.linear_white
        else:
            code_scene = hlg_inv_oetf(norm).astype(_F)
    hlg_g = 1.2 + 0.42 * np.log10(args.hlg_peak / 1000.0)

    def to_nits(r: np.ndarray) -> np.ndarray:
        if code_nits is not None:
            return code_nits32[r]
        if tc == TC_LINEAR:
            x = r.astype(_F) if r.dtype != _F else r.copy()
            np.maximum(x, 0, out=x)
            x *= _F(args.linear_white / dec.code_max)
            return x
        if tc == TC_PQ:
            return _pq_eotf32(r.astype(_F) / _F(dec.code_max))
        sc = code_scene[r] if code_scene is not None else _hlg_inv_oetf32(r.astype(_F) / _F(dec.code_max))
        ys = np.maximum(_dot3(sc, luma_src), _F(1e-12))
        f = np.power(ys, _F(hlg_g - 1.0))
        f *= _F(args.hlg_peak)
        sc *= f[..., None]
        return sc

    code_nits32 = code_nits.astype(_F) if code_nits is not None else None

    def ref_of(n: np.ndarray) -> np.ndarray:
        return _dot3(n, luma_src) if args.mode == "luminance" else _max3(n)

    # ---- 源峰值（类似 MaxCLL）：网格降采样统计 ----
    if args.source_peak == "auto":
        s = max(1, int(np.sqrt(h * w / 2_000_000)))
        stat = ref_of(to_nits(raw[::s, ::s]))
        Lw = float(np.percentile(stat, args.peak_percentile))
        del stat
    else:
        Lw = float(args.source_peak)

    Lmax = float(args.target_peak or args.sdr_white)
    Lw = max(Lw, 1e-3)
    eetf = None if Lw <= Lmax else EETF(Lw=Lw, Lmax=Lmax, Lb=0.0,
                                        Lmin=args.target_black, knee_offset=args.knee_offset)

    print(f"输入   : {dec.note} | {TRANSFER_NAMES[tc]} | {PRIMARY_NAMES[pr]} | {w}x{h}")
    print(f"源峰值 Lw = {Lw:.1f} nit（{f'自动, P{args.peak_percentile:g}' if args.source_peak == 'auto' else '手动'}）"
          f" | 目标峰值 Lmax = {Lmax:.1f} nit | SDR 白 = {args.sdr_white:.1f} nit")
    if eetf is None:
        print("源峰值不超过目标峰值：无需滚降，仅做线性缩放与色域转换。")
    else:
        print(f"EETF   : KS = {eetf.ks:.4f}（拐点 ≈ {eetf.knee_nits:.1f} nit，"
              f"相当于 SDR {eetf.knee_nits / args.sdr_white * 100:.0f}% 白），模式 = {args.mode}")

    # ---- 色调映射：整数 PQ/线性输入 + maxrgb/rgb 模式可完全由码值查表（精确，无插值）----
    gain_lut = None
    code_rel = code_gain = code_mapped = None
    if eetf is not None:
        def gain_fn(L):
            L = np.asarray(L, dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                g = eetf(L) / L
            return np.where(L > 0, g, 1.0) * inv_white
        if code_nits is not None and args.mode in ("maxrgb", "rgb"):
            if args.mode == "maxrgb":
                code_rel = code_nits32
                code_gain = gain_fn(code_nits).astype(_F)      # 增益按“最大通道码值”索引
            else:
                code_mapped = (eetf(code_nits) * inv_white).astype(_F)
        else:
            gain_lut = _LogLUT(gain_fn)
    elif code_nits is not None:
        code_mapped = (code_nits * inv_white).astype(_F)

    def tone(r: np.ndarray) -> np.ndarray:
        if code_mapped is not None:
            return code_mapped[r]
        if code_gain is not None:
            o = code_rel[r]
            o *= code_gain[_max3(r)][..., None]
            return o
        n = to_nits(r)
        if gain_lut is None:
            n *= _F(inv_white)
        elif args.mode == "rgb":
            n *= gain_lut(n)
        else:
            n *= gain_lut(ref_of(n))[..., None]
        return n

    # ---- 分块并行：色调映射 → 色域转换 → 色域映射 → sRGB 量化 ----
    bits = 8 if os.path.splitext(args.output)[1].lower() in (".jpg", ".jpeg", ".webp") else args.bits
    dst = np.empty((h, w, 3), dtype=np.uint8 if bits == 8 else np.uint16)
    noise = None
    if bits == 8:   # 抖动：±0.5 级均匀噪声，三通道共用，按 256 行周期平铺
        noise = (np.random.default_rng(0).random((_STRIP, w), dtype=_F) - _F(0.5))[..., None]

    def run(y0: int) -> None:
        y1 = min(h, y0 + _STRIP)
        o = tone(raw[y0:y1])
        if Mt is not None:
            o = (o.reshape(-1, 3) @ Mt).reshape(o.shape)
        if args.gamut_map == "clip":
            np.clip(o, 0, 1, out=o)
        else:
            bad = np.flatnonzero((_min3(o) < 0) | (_max3(o) > 1))
            if bad.size:
                flat = o.reshape(-1, 3)
                flat[bad] = gamut_fit(flat[bad], luma_out, "compress")
        if bits == 8:
            o *= _F(65535)
            o += _F(0.5)
            v = _SRGB8[o.astype(np.uint16)]
            v += noise[: y1 - y0]
            np.clip(v, 0, _F(255.99), out=v)
            dst[y0:y1] = v
        else:
            v = _srgb_oetf32(o)
            v *= _F(65535)
            v += _F(0.5)
            dst[y0:y1] = v

    starts = range(0, h, _STRIP)
    nt = min(_threads(args.threads), len(starts))
    if nt <= 1:
        for y0 in starts:
            run(y0)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(nt) as ex:
            list(ex.map(run, starts))
    del raw, dec.raw

    if dec.orientation != 1:
        dst = np.ascontiguousarray(apply_orientation(dst, dec.orientation))
    encode(dst, bits, args, out_pr, dec.exif)


def _exif_clean(exif: bytes | None) -> bytes | None:
    """保留 EXIF，但方向置 1、删除 MakerNote（避免携带厂商 HDR 增益提示）"""
    if not exif:
        return None
    try:
        from PIL import Image
        if not (exif.startswith(b"Exif\0\0") or exif[:4] in (b"II*\0", b"MM\0*")):
            pos = min([i for i in (exif.find(b"II*\0"), exif.find(b"MM\0*")) if i >= 0], default=-1)
            if pos < 0:
                return None
            exif = exif[pos:]
        ex = Image.Exif()
        ex.load(exif)
        ex[0x0112] = 1
        ifd = ex.get_ifd(0x8769)
        ifd.pop(0x927C, None)
        return ex.tobytes()
    except Exception:
        return None


def _adler32_combine(a1: int, a2: int, len2: int) -> int:
    """zlib adler32_combine 的 Python 实现：合并两段数据的 Adler-32"""
    base = 65521
    rem = len2 % base
    s1 = a1 & 0xFFFF
    s2 = (rem * s1) % base
    s1 += (a2 & 0xFFFF) + base - 1
    s2 += (a1 >> 16) + (a2 >> 16) + base - rem
    if s1 >= base: s1 -= base
    if s1 >= base: s1 -= base
    if s2 >= base << 1: s2 -= base << 1
    if s2 >= base: s2 -= base
    return s1 | (s2 << 16)


def _png_chunk(f, typ: bytes, data: bytes) -> None:
    f.write(struct.pack(">I", len(data)) + typ)
    f.write(data)
    f.write(struct.pack(">I", zlib.crc32(data, zlib.crc32(typ)) & 0xFFFFFFFF))


def write_png(path: str, q: np.ndarray, level: int = 6, icc: bytes | None = None,
              exif: bytes | None = None, threads: int = 0) -> None:
    """多线程 PNG 写入（RGB，8/16 bit）：
    Paeth 滤波向量化计算；各行块独立 raw-deflate（SYNC_FLUSH 衔接，同 pigz），Adler-32 分块后合并。"""
    h, w, c = q.shape
    depth = q.dtype.itemsize * 8
    rows = (q.astype(">u2") if depth == 16 else q).view(np.uint8).reshape(h, w * c * q.dtype.itemsize)
    bpp = c * q.dtype.itemsize
    starts = list(range(0, h, _STRIP))

    def job(y0: int):
        y1 = min(h, y0 + _STRIP)
        x = rows[y0:y1].astype(np.int16)
        up = np.empty_like(x)
        up[1:] = x[:-1]
        up[0] = rows[y0 - 1] if y0 > 0 else 0
        a = np.zeros_like(x); a[:, bpp:] = x[:, :-bpp]                 # 左
        cc = np.zeros_like(x); cc[:, bpp:] = up[:, :-bpp]              # 左上
        p = a + up - cc
        pa, pb, pc = np.abs(p - a), np.abs(p - up), np.abs(p - cc)
        pred = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, up, cc))
        filt = np.empty((y1 - y0, rows.shape[1] + 1), dtype=np.uint8)
        filt[:, 0] = 4                                                  # Paeth
        filt[:, 1:] = (x - pred).astype(np.uint8)
        raw = filt.tobytes()
        co = zlib.compressobj(level, zlib.DEFLATED, -15, 9)
        body = co.compress(raw) + co.flush(zlib.Z_FINISH if y1 == h else zlib.Z_SYNC_FLUSH)
        return body, zlib.adler32(raw), len(raw)

    nt = min(_threads(threads), len(starts))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        _png_chunk(f, b"IHDR", struct.pack(">IIBBBBB", w, h, depth, 2, 0, 0, 0))
        if icc:
            _png_chunk(f, b"iCCP", b"ICC profile\0\0" + zlib.compress(icc))
        if exif:
            _png_chunk(f, b"eXIf", exif[6:] if exif.startswith(b"Exif\0\0") else exif)
        adler, first = 1, True
        if nt <= 1:
            results = map(job, starts)
        else:
            from concurrent.futures import ThreadPoolExecutor
            ex = ThreadPoolExecutor(nt)
            results = ex.map(job, starts)        # 按顺序产出，边压缩边写
        try:
            for body, a32, n in results:
                adler = _adler32_combine(adler, a32, n)
                if first:
                    body = b"\x78\x9c" + body
                    first = False
                _png_chunk(f, b"IDAT", body)
        finally:
            if nt > 1:
                ex.shutdown()
        _png_chunk(f, b"IDAT", struct.pack(">I", adler))
        _png_chunk(f, b"IEND", b"")


def encode(q: np.ndarray, bits: int, args, out_pr: int, exif: bytes | None) -> None:
    """q: 已量化的 sRGB 码值（uint8 / uint16）"""
    path = args.output
    ext = os.path.splitext(path)[1].lower()
    icc = build_icc(out_pr, "sRGB IEC61966-2.1 (hdr2sdr)" if out_pr == 1 else "Display P3 (hdr2sdr)")
    exif = None if args.no_exif else _exif_clean(exif)
    if bits != args.bits:
        print("提示：JPEG/WebP 只支持 8 bit，已改为 8 bit")

    if bits == 16:
        if ext == ".png":
            write_png(path, q, args.png_level, icc, exif, args.threads)
        elif ext in (".tif", ".tiff"):
            import tifffile
            tifffile.imwrite(path, q, photometric="rgb", compression="zlib", maxworkers=_threads(args.threads),
                             rowsperstrip=64, extratags=[(34675, 7, len(icc), icc, True)])
        else:
            raise SystemExit("错误：16 bit 输出仅支持 .png / .tif")
    elif ext == ".png":
        write_png(path, q, args.png_level, icc, exif, args.threads)
    else:
        from PIL import Image
        im = Image.fromarray(q, "RGB")
        kw = {"icc_profile": icc}
        if exif:
            kw["exif"] = exif
        if ext in (".jpg", ".jpeg"):
            im.save(path, quality=args.quality, subsampling=0 if args.quality >= 90 else 2,
                    optimize=args.optimize, **kw)
        elif ext == ".webp":
            im.save(path, quality=args.quality, method=4, **kw)
        elif ext in (".tif", ".tiff"):
            im.save(path, compression="tiff_adobe_deflate", **kw)
        else:
            raise SystemExit(f"错误：不支持的输出格式 {ext}")
    print(f"输出   : {path}（{'sRGB' if out_pr == 1 else 'Display P3'}，{bits} bit，无 gain map / HDR 元数据）")


def _utf8_console():
    """Windows 下输出被重定向时控制台编码可能是 cp1252/cp936，避免打印中文时报错"""
    for s in (sys.stdout, sys.stderr):
        try:
            if s is not None and (s.encoding or "").lower().replace("-", "") != "utf8":
                s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_INPUT_EXTS = (".avif", ".heic", ".heif", ".hif", ".jxl", ".png", ".tif", ".tiff", ".exr")


def _launched_by_double_click() -> bool:
    """Windows 资源管理器双击启动时，控制台只属于本程序（onefile 打包时还有一个引导进程）"""
    if os.name != "nt":
        return False
    try:
        import ctypes
        buf = (ctypes.c_uint * 4)()
        n = ctypes.windll.kernel32.GetConsoleProcessList(buf, 4)
        return 1 <= n <= (2 if getattr(sys, "frozen", False) else 1)   # n=0：没有控制台（如 CI）
    except Exception:
        return False


def _pick_files() -> list[str]:
    """没有给出输入文件时：弹出文件选择框；不可用时改为在控制台输入（可直接把文件拖进窗口）"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        pattern = " ".join("*" + e for e in _INPUT_EXTS)
        files = filedialog.askopenfilenames(
            title="选择要转换的 HDR 照片（可多选）",
            filetypes=[("HDR 照片", pattern), ("所有文件", "*.*")])
        root.destroy()
        return list(files)
    except Exception:
        pass
    print("请把 HDR 照片拖进本窗口（或输入路径），多个文件用空格分隔，然后按回车：")
    try:
        line = input("> ").strip()
    except EOFError:
        return []
    import shlex
    try:
        return shlex.split(line, posix=(os.name != "nt"))
    except ValueError:
        return [line]


def main(argv=None):
    _utf8_console()
    interactive = argv is None and len(sys.argv) <= 1
    pause = interactive or (argv is None and _launched_by_double_click())
    try:
        return _main(argv, interactive)
    finally:
        if pause:
            try:
                input("\n按回车键退出…")
            except EOFError:
                pass


def _main(argv, interactive):
    p = argparse.ArgumentParser(
        description="使用 ITU-R BT.2390 EETF 将 HDR 照片转换为 SDR（完全忽略 gain map）。"
                    "不带参数运行（如双击）会弹出文件选择框；也可把照片直接拖到程序图标上。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", nargs="*", help="一个或多个 HDR 照片")
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-o", "--output", help="输出路径（.jpg/.png/.tif/.webp），仅限单个输入；默认 <输入名>_sdr.jpg")
    p.add_argument("--format", choices=["jpg", "png", "tif", "webp"], default="jpg",
                   help="未指定 -o 时的输出格式")
    g = p.add_argument_group("色调映射")
    g.add_argument("--sdr-white", type=float, default=203.0,
                   help="映射为 SDR 100%% 白的 HDR 亮度（nit）。BT.2408 参考白为 203")
    g.add_argument("--target-peak", type=float, default=None,
                   help="EETF 目标峰值 Lmax（nit），默认等于 --sdr-white，即 SDR 峰值 = SDR 白")
    g.add_argument("--target-black", type=float, default=0.0, help="EETF 目标黑位 Lmin（nit），>0 时启用暗部 toe")
    g.add_argument("--source-peak", default="auto", help="源峰值 Lw（nit）或 auto（按内容统计，类似 MaxCLL）")
    g.add_argument("--peak-percentile", type=float, default=99.99, help="auto 模式下用于求源峰值的百分位，100 = 绝对最大值")
    g.add_argument("--knee-offset", type=float, default=0.5,
                   help="拐点偏移 k：KS=(1+k)·maxLum−k。0.5 为 BT.2390 规范值；越大拐点越低、滚降越柔")
    g.add_argument("--mode", choices=["maxrgb", "luminance", "rgb"], default="maxrgb",
                   help="EETF 作用对象：maxrgb（保色相、不溢出）/ luminance（亮度比例）/ rgb（逐通道，会偏色）")
    g = p.add_argument_group("输入解释")
    g.add_argument("--transfer", choices=["pq", "hlg", "linear"], default=None, help="覆盖自动识别的传递函数")
    g.add_argument("--primaries", choices=["bt2020", "p3", "bt709"], default=None, help="覆盖自动识别的色彩原色")
    g.add_argument("--hlg-peak", type=float, default=1000.0, help="HLG 假定的显示峰值（nit），用于 OOTF")
    g.add_argument("--linear-white", type=float, default=203.0, help="线性输入中 1.0 对应的亮度（nit）")
    g = p.add_argument_group("输出")
    g.add_argument("--gamut", choices=["srgb", "p3"], default="srgb", help="输出色域（均使用 sRGB 传递曲线）")
    g.add_argument("--gamut-map", choices=["compress", "clip"], default="compress", help="超色域处理")
    g.add_argument("--bits", type=int, choices=[8, 16], default=8)
    g.add_argument("--quality", type=int, default=95, help="JPEG/WebP 质量")
    g.add_argument("--no-exif", action="store_true", help="不保留 EXIF")
    g.add_argument("--png-level", type=int, default=6, help="PNG 压缩级别 0–9（越小越快、文件越大）")
    g.add_argument("--no-optimize", dest="optimize", action="store_false",
                   help="JPEG 不做霍夫曼表优化（快约 20%%，文件稍大）")
    p.add_argument("-j", "--threads", type=int, default=0, help="线程数，0 = 全部 CPU 核心")
    args = p.parse_args(argv)
    inputs = list(args.input)
    if not inputs:
        print(f"hdr2sdr {__version__} — BT.2390 EETF HDR → SDR")
        inputs = _pick_files()
    inputs = [s.strip().strip('"').strip("'") for s in inputs if s.strip()]
    if not inputs:
        print("没有选择文件。命令行用法见 hdr2sdr --help")
        return 1
    if args.output and len(inputs) > 1:
        p.error("-o 只能用于单个输入文件；多个文件时会在原文件旁生成 <名称>_sdr.<格式>")

    failed = 0
    for i, src in enumerate(inputs, 1):
        if len(inputs) > 1:
            print(f"\n[{i}/{len(inputs)}] {src}")
        a = argparse.Namespace(**vars(args))
        a.input = src
        a.output = args.output or os.path.splitext(src)[0] + "_sdr." + args.format
        try:
            if not os.path.isfile(src):
                raise SystemExit(f"错误：找不到文件 {src}")
            process(a)
        except SystemExit as e:
            failed += 1
            print(e.code if isinstance(e.code, str) else f"失败（{e.code}）", file=sys.stderr)
        except Exception as e:                      # 单个文件出错不影响其余文件
            failed += 1
            print(f"错误：{type(e).__name__}: {e}", file=sys.stderr)
    if len(inputs) > 1:
        print(f"\n完成：成功 {len(inputs) - failed} 个，失败 {failed} 个")
    if failed and len(inputs) == 1 and not interactive:
        raise SystemExit(1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
