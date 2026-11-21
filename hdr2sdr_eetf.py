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

__version__ = "1.0.0"

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
    pixels: np.ndarray          # float32, HxWx3, 归一化码值 [0,1]（线性输入则为线性值）
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


def _norm_uint(a: np.ndarray, bits: int | None) -> np.ndarray:
    if a.dtype.kind == "f":
        return a.astype(np.float32)
    if bits is None:
        bits = 8 if a.dtype == np.uint8 else 16
    return (a.astype(np.float32) / float((1 << bits) - 1))


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


def decode(path: str) -> Decoded:
    ext = os.path.splitext(path)[1].lower()
    data = open(path, "rb").read()

    if ext in (".heic", ".heif", ".hif"):
        import pillow_heif
        hf = pillow_heif.open_heif(path, convert_hdr_to_8bit=False)  # 只取主图；辅助图/增益图不读
        bd = int(getattr(hf, "bit_depth", 8) or 8)
        arr = np.asarray(hf)
        if arr.dtype == np.uint16:   # pillow-heif 把 10/12bit 左移到 16bit 高位
            arr = (arr >> (16 - bd)).astype(np.float32) / float((1 << bd) - 1)
        else:
            arr = arr.astype(np.float32) / 255.0
        nclx = hf.info.get("nclx_profile") or {}
        tc, pr = nclx.get("transfer_characteristics"), nclx.get("color_primaries")
        if tc is None:
            c = _cicp_from_icc(hf.info.get("icc_profile"))
            if c:
                pr, tc = c
        return Decoded(_rgb3(arr), tc, pr, hf.info.get("exif"), 1, "HEIF (libheif 已应用旋转)")

    import imagecodecs as ic
    if ext == ".avif":
        info = ic.avif_decode(data, _info=True)
        arr = _norm_uint(ic.avif_decode(data), info.get("bitspersample"))
        tc, pr = info.get("transfer"), info.get("primaries")
        if (tc in (None, 2)) and info.get("iccprofile"):
            c = _cicp_from_icc(info["iccprofile"])
            if c:
                pr, tc = c
        exif, orient = _pil_side_info(path)   # libavif 不应用 irot/imir，这里自行处理
        return Decoded(_rgb3(arr), int(tc) if tc is not None else None,
                       int(pr) if pr is not None else None, exif, orient, "AVIF")

    if ext == ".jxl":
        info = ic.jpegxl_decode(data, _info=True)
        arr = _norm_uint(ic.jpegxl_decode(data), info.get("bitspersample"))  # libjxl 已应用方向
        tc, pr = info.get("transfer"), info.get("primaries")
        if tc in (None, 2) or pr in (None, 2):
            c = _cicp_from_icc(info.get("iccprofile"))
            if c:
                pr, tc = c
        return Decoded(_rgb3(arr), int(tc) if tc is not None else None,
                       int(pr) if pr is not None else None, None, 1, "JPEG XL")

    if ext == ".png":
        info = ic.png_decode(data, _info=True)
        arr = _norm_uint(ic.png_decode(data), info.get("bit_depth"))
        tc, pr = info.get("transfer"), info.get("primaries")
        if tc is None:
            c = _cicp_from_icc(info.get("iccprofile"))
            if c:
                pr, tc = c
        exif, orient = _pil_side_info(path)
        return Decoded(_rgb3(arr), int(tc) if tc is not None else None,
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
        return Decoded(_rgb3(_norm_uint(arr, None)), tc, pr, exif, orient, "TIFF")

    if ext == ".exr":
        arr = ic.exr_decode(data).astype(np.float32)
        return Decoded(_rgb3(arr), TC_LINEAR, None, None, 1, "OpenEXR（线性）")

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
def to_display_nits(code: np.ndarray, tc: int, pr: int, args) -> np.ndarray:
    if tc == TC_PQ:
        return pq_eotf(code.astype(np.float64)).astype(np.float32)
    if tc == TC_HLG:
        luma = rgb_to_xyz_matrix(pr)[1]
        return hlg_to_display_nits(code.astype(np.float64), args.hlg_peak, luma).astype(np.float32)
    if tc == TC_LINEAR:
        return np.maximum(code, 0).astype(np.float32) * np.float32(args.linear_white)
    raise ValueError(tc)


def gamut_fit(rgb: np.ndarray, luma: np.ndarray, method: str) -> np.ndarray:
    """把线性 RGB 放进 [0,1]：compress = 沿亮度轴向灰色收拢（保持亮度与色相），clip = 硬裁切"""
    if method == "clip":
        return np.clip(rgb, 0.0, 1.0)
    y = np.clip(rgb @ luma, 0.0, 1.0)[..., None]
    d = rgb - y
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lo = np.where(rgb < 0, y / np.where(d < 0, -d, 1), np.inf)
        t_hi = np.where(rgb > 1, (1 - y) / np.where(d > 0, d, 1), np.inf)
    t = np.clip(np.minimum(t_lo, t_hi).min(axis=-1, keepdims=True), 0.0, 1.0)
    return np.clip(y + t * d, 0.0, 1.0)


def process(args) -> None:
    dec = decode(args.input)
    tc = args.transfer if args.transfer is not None else dec.transfer
    pr = args.primaries if args.primaries is not None else dec.primaries

    tc_map = {"pq": TC_PQ, "hlg": TC_HLG, "linear": TC_LINEAR}
    if isinstance(tc, str):
        tc = tc_map[tc]
    pr_map = {"bt2020": 9, "p3": 12, "bt709": 1}
    if isinstance(pr, str):
        pr = pr_map[pr]
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

    pix = apply_orientation(dec.pixels, dec.orientation)
    h, w = pix.shape[:2]
    out_pr = {"srgb": 1, "p3": 12}[args.gamut]
    luma_src = rgb_to_xyz_matrix(pr)[1].astype(np.float32)
    luma_out = rgb_to_xyz_matrix(out_pr)[1].astype(np.float32)
    M = conversion_matrix(pr, out_pr).astype(np.float32)

    # ---- 第一遍：计算源峰值（类似 MaxCLL） ----
    step = max(1, (h * w) // 4_000_000)          # 大图降采样统计，避免内存过大
    sample = to_display_nits(pix.reshape(-1, 3)[::step], tc, pr, args)
    if args.mode == "luminance":
        stat = sample @ luma_src
    else:
        stat = sample.max(axis=-1)
    if args.source_peak == "auto":
        Lw = float(np.percentile(stat, args.peak_percentile))
    else:
        Lw = float(args.source_peak)
    del sample, stat

    Lmax = float(args.target_peak or args.sdr_white)
    Lw = max(Lw, 1e-3)
    passthrough = Lw <= Lmax
    eetf = None if passthrough else EETF(Lw=Lw, Lmax=Lmax, Lb=0.0,
                                          Lmin=args.target_black, knee_offset=args.knee_offset)

    print(f"输入   : {dec.note} | {TRANSFER_NAMES[tc]} | {PRIMARY_NAMES[pr]} | {w}x{h}")
    print(f"源峰值 Lw = {Lw:.1f} nit（{f'自动, P{args.peak_percentile:g}' if args.source_peak == 'auto' else '手动'}）"
          f" | 目标峰值 Lmax = {Lmax:.1f} nit | SDR 白 = {args.sdr_white:.1f} nit")
    if passthrough:
        print("源峰值不超过目标峰值：无需滚降，仅做线性缩放与色域转换。")
    else:
        print(f"EETF   : KS = {eetf.ks:.4f}（拐点 ≈ {eetf.knee_nits:.1f} nit，"
              f"相当于 SDR {eetf.knee_nits / args.sdr_white * 100:.0f}% 白），模式 = {args.mode}")

    # ---- 第二遍：分块色调映射 ----
    out = np.empty((h, w, 3), dtype=np.float32)
    rows = max(1, 2_000_000 // max(w, 1))
    for y0 in range(0, h, rows):
        blk = to_display_nits(pix[y0:y0 + rows], tc, pr, args)
        blk = np.maximum(blk, 0.0)
        if eetf is not None:
            if args.mode == "rgb":
                blk = eetf(blk.astype(np.float64)).astype(np.float32)
            else:
                ref = blk.max(axis=-1) if args.mode == "maxrgb" else blk @ luma_src
                ref = np.maximum(ref, 1e-6)
                mapped = eetf(ref.astype(np.float64)).astype(np.float32)
                blk = blk * (mapped / ref)[..., None]
        rel = blk / np.float32(args.sdr_white)            # 1.0 = SDR 白
        rel = rel @ M.T                                   # 转到输出色域
        out[y0:y0 + rows] = gamut_fit(rel, luma_out, args.gamut_map)

    encode(out, args, out_pr, dec.exif)


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


def _png_insert_chunks(png: bytes, chunks: list[tuple[bytes, bytes]]) -> bytes:
    i = 8 + 4 + 4 + 13 + 4                       # 跳过签名与 IHDR
    extra = b""
    for typ, data in chunks:
        extra += struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
    return png[:i] + extra + png[i:]


def encode(lin: np.ndarray, args, out_pr: int, exif: bytes | None) -> None:
    enc = srgb_oetf(lin)
    path = args.output
    ext = os.path.splitext(path)[1].lower()
    icc = build_icc(out_pr, "sRGB IEC61966-2.1 (hdr2sdr)" if out_pr == 1 else "Display P3 (hdr2sdr)")
    exif = None if args.no_exif else _exif_clean(exif)
    bits = args.bits
    if ext in (".jpg", ".jpeg", ".webp") and bits != 8:
        print("提示：JPEG/WebP 只支持 8 bit，已改为 8 bit")
        bits = 8

    if bits == 16:
        q = np.round(enc * 65535.0).astype(np.uint16)
        if ext == ".png":
            import imagecodecs as ic
            data = ic.png_encode(q, level=6)
            chunks = [(b"iCCP", b"hdr2sdr\0\0" + zlib.compress(icc))]
            if exif:
                chunks.append((b"eXIf", exif[6:] if exif.startswith(b"Exif\0\0") else exif))
            open(path, "wb").write(_png_insert_chunks(data, chunks))
        elif ext in (".tif", ".tiff"):
            import tifffile
            tifffile.imwrite(path, q, photometric="rgb", compression="zlib",
                             extratags=[(34675, 7, len(icc), icc, True)])
        else:
            raise SystemExit("错误：16 bit 输出仅支持 .png / .tif")
    else:
        from PIL import Image
        # 8 bit 量化加轻微有序抖动，减少滚降区域的色带
        rng = np.random.default_rng(0)
        noise = (rng.random(enc.shape[:2], dtype=np.float32)[..., None] - 0.5) / 255.0
        q = np.clip(np.round((enc + noise) * 255.0), 0, 255).astype(np.uint8)
        im = Image.fromarray(q, "RGB")
        kw = {"icc_profile": icc}
        if exif:
            kw["exif"] = exif
        if ext in (".jpg", ".jpeg"):
            im.save(path, quality=args.quality, subsampling=0 if args.quality >= 90 else 2, optimize=True, **kw)
        elif ext == ".webp":
            im.save(path, quality=args.quality, **kw)
        elif ext == ".png":
            im.save(path, optimize=True, **kw)
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


def main(argv=None):
    _utf8_console()
    p = argparse.ArgumentParser(
        description="使用 ITU-R BT.2390 EETF 将 HDR 照片转换为 SDR（完全忽略 gain map）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input")
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-o", "--output", help="输出路径（.jpg/.png/.tif/.webp），默认 <输入名>_sdr.jpg")
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
    args = p.parse_args(argv)
    if not args.output:
        args.output = os.path.splitext(args.input)[0] + "_sdr.jpg"
    process(args)


if __name__ == "__main__":
    main()
