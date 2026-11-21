import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hdr2sdr_eetf as h  # noqa: E402
import scene  # noqa: E402


@pytest.fixture(scope="session")
def images(tmp_path_factory):
    return scene.write_all(str(tmp_path_factory.mktemp("img")))


def run(*argv):
    h.main([str(a) for a in argv])


def read_linear(path):
    import imagecodecs as ic
    a = ic.png_decode(open(path, "rb").read()).astype(np.float64)
    a /= 65535.0 if a.max() > 255 else 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


# ---------------------------------------------------------------- EETF 数学
@pytest.mark.parametrize("Lw,Lmax,k", [(1000, 203, 0.5), (4000, 203, 0.5), (1000, 100, 0.5), (1600, 203, 1.0), (300, 203, 0.5)])
def test_eetf_properties(Lw, Lmax, k):
    e = h.EETF(Lw=Lw, Lmax=Lmax, knee_offset=k)
    x = np.geomspace(0.01, Lw, 20001)
    y = e(x)
    below = x < e.knee_nits * 0.999
    assert np.max(np.abs(y[below] / x[below] - 1)) < 1e-9          # 拐点以下恒等
    assert np.all(np.diff(y) >= -1e-9)                               # 单调
    assert abs(e(np.array([Lw]))[0] - Lmax) < 1e-6                   # 峰值映射
    assert abs(e(np.array([Lw * 3]))[0] - Lmax) < 1e-6               # 超出峰值不越界


def test_eetf_toe():
    e = h.EETF(Lw=1000, Lmax=203, Lmin=0.1)
    assert abs(e(np.array([0.0]))[0] - 0.1) < 1e-6


def test_pq_roundtrip():
    x = np.geomspace(1e-3, 10000, 1000)
    assert np.allclose(h.pq_eotf(h.pq_inv_eotf(x)), x, rtol=1e-6)


# ---------------------------------------------------------------- ICC
@pytest.mark.parametrize("prim", [1, 12])
def test_icc_matches_matrix(prim):
    from PIL import Image, ImageCms
    prof = ImageCms.ImageCmsProfile(io.BytesIO(h.build_icc(prim, "t")))
    t = ImageCms.buildTransform(prof, ImageCms.createProfile("sRGB"), "RGB", "RGB")
    test = np.array([[[128, 128, 128], [200, 60, 40], [40, 180, 90], [255, 255, 255]]], np.uint8)
    r = np.asarray(ImageCms.applyTransform(Image.fromarray(test), t)).astype(int)
    x = test / 255.0
    xl = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    exp = np.round(h.srgb_oetf(np.clip(xl @ h.conversion_matrix(prim, 1).T, 0, 1)) * 255)
    assert np.max(np.abs(r - exp)) <= 1


# ---------------------------------------------------------------- 端到端
@pytest.mark.parametrize("name", ["pq.avif", "pq.jxl", "pq.png", "pq.heic", "hlg.avif", "linear.exr"])
def test_formats_convert(images, tmp_path, name):
    if name not in images:
        pytest.skip("当前平台无法生成该格式")
    out = tmp_path / "o.png"
    run(images[name], "-o", out, "--bits", "16")
    lin = read_linear(out)
    assert lin.shape == (scene.H, scene.W, 3)
    assert abs(lin[185:200, 235:255].mean() - 0.18) < 0.01           # 中灰低于拐点，应保持不变
    assert lin[:40, -5:].mean() > 0.97                                # 最亮处到达 SDR 白
    ramp = lin[20, :, 1]
    assert np.all(np.diff(ramp) > -0.01)                              # 灰阶单调


def test_matches_reference(images, tmp_path):
    out = tmp_path / "o.png"
    run(images["pq.png"], "-o", out, "--bits", "16", "--source-peak", "4000")
    nits = scene.scene_nits().astype(np.float64)
    e = h.EETF(Lw=4000, Lmax=203)
    m = np.maximum(nits.max(-1), 1e-6)
    ref = nits * (e(m) / m)[..., None] / 203
    ref = h.gamut_fit(ref @ h.conversion_matrix(9, 1).T, h.rgb_to_xyz_matrix(1)[1], "compress")
    d = np.abs(h.srgb_oetf(read_linear(out)) - h.srgb_oetf(ref)) * 255
    assert d.mean() < 0.5 and np.percentile(d, 99) < 1.5


@pytest.mark.parametrize("ext", [".jpg", ".webp", ".tif", ".png"])
def test_output_formats(images, tmp_path, ext):
    out = tmp_path / ("o" + ext)
    run(images["pq.avif"], "-o", out, "--gamut", "p3")
    from PIL import Image
    with Image.open(out) as im:
        assert im.size == (scene.W, scene.H)
        assert im.info.get("icc_profile")


def test_jpeg_has_no_gain_map(images, tmp_path):
    out = tmp_path / "o.jpg"
    run(images["pq.avif"], "-o", out)
    d = out.read_bytes()
    for token in (b"MPF", b"hdrgm", b"urn:iso:std:iso:ts:21496", b"GainMap"):
        assert token not in d
    assert d.count(b"\xff\xd8\xff") == 1


@pytest.mark.parametrize("mode", ["luminance", "rgb"])
def test_modes(images, tmp_path, mode):
    out = tmp_path / "o.png"
    run(images["pq.avif"], "-o", out, "--mode", mode)
    assert out.exists()


def test_refuses_sdr(images, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run(images["sdr.avif"], "-o", tmp_path / "o.jpg")
    assert "gain map" in capsys.readouterr().err


def test_batch_continues_after_error(images, tmp_path, capsys):
    import shutil
    a = tmp_path / "a.avif"
    b = tmp_path / "b.png"
    shutil.copy(images["pq.avif"], a)
    shutil.copy(images["pq.png"], b)
    rc = h.main([str(a), str(tmp_path / "missing.avif"), str(b), "--format", "png"])
    assert rc == 1
    assert (tmp_path / "a_sdr.png").exists() and (tmp_path / "b_sdr.png").exists()
    assert "成功 2 个，失败 1 个" in capsys.readouterr().out


def test_output_flag_single_only(images, tmp_path):
    with pytest.raises(SystemExit):
        run(images["pq.avif"], images["pq.png"], "-o", tmp_path / "o.jpg")


def test_refuses_jpeg(tmp_path):
    p = tmp_path / "x.jpg"
    from PIL import Image
    Image.new("RGB", (8, 8)).save(p)
    with pytest.raises(SystemExit):
        run(p, "-o", tmp_path / "o.jpg")


def test_tiff_orientation(tmp_path):
    import tifffile
    from PIL import Image
    a = np.zeros((40, 80, 3), np.float32)
    a[:10, :20] = 5.0
    src = tmp_path / "o.tif"
    tifffile.imwrite(src, a, photometric="rgb", extratags=[(274, 3, 1, 6, True)])
    out = tmp_path / "o.png"
    run(src, "-o", out)
    o = np.asarray(Image.open(out))
    assert o.shape[:2] == (80, 40)
    yx = np.argwhere(o[..., 0] > 200).mean(0)
    assert yx[0] < 20 and yx[1] > 20                                 # 旋转 90° 后亮块在右上
