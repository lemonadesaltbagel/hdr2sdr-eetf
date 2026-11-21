"""生成合成 HDR 测试场景（PQ / HLG / 线性），供测试与 CI 冒烟测试使用。

用法：python tests/scene.py <输出目录>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hdr2sdr_eetf as h  # noqa: E402

H, W = 240, 360


def scene_nits() -> np.ndarray:
    """BT.2020 线性、绝对亮度（nit）的测试场景"""
    img = np.zeros((H, W, 3))
    img[:40] = np.geomspace(0.05, 4000, W)[None, :, None]           # 灰阶 0.05→4000 nit
    yy, xx = np.mgrid[0:H, 0:W]
    img[40:] = (np.array([0.35, 0.55, 1.0]) * (300 * (1 - (yy - 40) / 200.0))[..., None])[40:]
    glow = np.exp(-((xx - 280) ** 2 + (yy - 85) ** 2) / (2 * 35.0 ** 2))
    img[40:] += glow[40:, :, None] * np.array([1.0, 0.9, 0.7]) * 1500
    img[(xx - 280) ** 2 + (yy - 85) ** 2 < 18 ** 2] = np.array([1.0, 0.97, 0.9]) * 4000
    for i, c in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]):
        for j, L in enumerate((50, 203, 800)):
            y0, x0 = 150 + j * 28, 10 + i * 35
            img[y0:y0 + 25, x0:x0 + 30] = np.array(c, float) * L
    img[180:205, 230:260] = 0.18 * 203                               # 中灰
    return img.astype(np.float32)


def write_all(out: str) -> dict:
    import imagecodecs as ic
    import pillow_heif

    os.makedirs(out, exist_ok=True)
    img = scene_nits()
    pq = h.pq_inv_eotf(img)
    q10 = np.round(pq * 1023).astype(np.uint16)
    files = {}

    def put(name, data):
        p = os.path.join(out, name)
        with open(p, "wb") as f:
            f.write(data)
        files[name] = p

    put("pq.avif", ic.avif_encode(q10, level=95, bitspersample=10, primaries=9, transfer=16, matrix=9))
    put("pq.jxl", ic.jpegxl_encode(q10, bitspersample=10, primaries=9, transfer=16, distance=0.5))
    put("pq.png", ic.png_encode(np.round(pq * 65535).astype(np.uint16), primaries=9, transfer=16, matrix=0))
    try:
        hf = pillow_heif.from_bytes(mode="RGB;16", size=(W, H), data=(q10 << 6).tobytes())
        hf.info["nclx_profile"] = {"color_primaries": 9, "transfer_characteristics": 16,
                                   "matrix_coefficients": 9, "full_range_flag": 1}
        p = os.path.join(out, "pq.heic")
        hf.save(p, quality=95, bit_depth=10)
        files["pq.heic"] = p
    except Exception as e:  # 某些平台的 pillow-heif 轮子没有 HEVC 编码器
        print("跳过 HEIC：", e)

    # HLG（1000 nit 母版）
    M = h.rgb_to_xyz_matrix(9)
    Lw, g = 1000.0, 1.2
    Yd = np.maximum(img @ M[1], 1e-9)[..., None]
    Ys = np.power(np.minimum(Yd / Lw, 1), 1 / g)
    scene = np.clip(img / Lw / np.power(np.maximum(Ys, 1e-9), g - 1), 0, 1)
    hlg = np.where(scene <= 1 / 12, np.sqrt(3 * scene),
                   h._HA * np.log(np.maximum(12 * scene - h._HB, 1e-9)) + h._HC)
    put("hlg.avif", ic.avif_encode(np.round(hlg * 1023).astype(np.uint16), level=95,
                                   bitspersample=10, primaries=9, transfer=18, matrix=9))
    lin709 = ((img / 203.0) @ h.conversion_matrix(9, 1).T).astype(np.float32)
    put("linear.exr", ic.exr_encode(lin709))
    put("sdr.avif", ic.avif_encode((h.srgb_oetf(np.clip(lin709, 0, 1)) * 255).astype(np.uint8),
                                   primaries=1, transfer=13, matrix=1))
    return files


if __name__ == "__main__":
    for k, v in write_all(sys.argv[1] if len(sys.argv) > 1 else "test_images").items():
        print(k, v)
