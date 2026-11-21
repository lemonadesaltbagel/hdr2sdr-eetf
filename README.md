# hdr2sdr-eetf

使用 **ITU-R BT.2390 EETF** 将 HDR 照片转换为 SDR 照片。

转换只使用 HDR 主图的像素（PQ / HLG / 线性），**完全不读取、不参考、不写出 gain map**。输出的是重新编码的普通 SDR 文件，只带 ICC（sRGB 或 Display P3）和可选的 EXIF。

## 下载

到 [Releases](../../releases/latest) 下载对应平台的单文件可执行程序，无需安装 Python：

| 平台 | 文件 |
|---|---|
| Windows x64 | `hdr2sdr-windows-x64.exe` |
| macOS Apple Silicon | `hdr2sdr-macos-arm64` |
| macOS Intel | `hdr2sdr-macos-x64` |
| Linux x64 | `hdr2sdr-linux-x64` |

macOS / Linux 首次使用需要加执行权限：`chmod +x hdr2sdr-*`。macOS 若提示“无法验证开发者”，执行 `xattr -d com.apple.quarantine hdr2sdr-macos-*`。

也可以用 pip 安装（Python ≥ 3.12，依赖 imagecodecs ≥ 2026.11.11）：

```bash
pip install git+https://github.com/<你的用户名>/hdr2sdr-eetf.git
hdr2sdr --help
```

## 用法

**双击运行**：弹出文件选择框，可一次选多张照片，结果保存在原文件旁（`<名称>_sdr.jpg`），处理完窗口会停住方便查看。
**拖放**：把一张或多张照片直接拖到程序图标上。

**命令行**：

```bash
hdr2sdr photo.avif                                  # → photo_sdr.jpg（sRGB）
hdr2sdr photo.heic -o out.png --bits 16 --gamut p3  # 16 bit Display P3 PNG
hdr2sdr merge.exr --linear-white 203                # 线性 EXR，1.0 = 203 nit
hdr2sdr photo.jxl --knee-offset 0.3                 # 拐点更高，保留更多中间调
hdr2sdr *.avif --format png                         # 批量转换，单张出错不影响其余
```

## 支持的格式

| 输入 | 说明 |
|---|---|
| `.avif` `.heic` `.heif` `.hif` `.jxl` `.png` | PQ / HLG 编码，自动从 CICP / nclx / ICC `cicp` 识别 |
| `.exr` `.tif` (浮点) | 线性数据，`--linear-white` 指定 1.0 对应的亮度 |

| 输出 | 位深 |
|---|---|
| `.jpg` `.webp` | 8 bit |
| `.png` `.tif` | 8 / 16 bit |

**主图本身是 SDR 的文件会被拒绝**，例如 iPhone 的 HEIC、Ultra HDR JPEG：这类文件的 HDR 信息只存在于 gain map 中，而本工具按设计不读取 gain map。

## 处理流程

1. **解码**：只解码主图，按 PQ EOTF 或 HLG OOTF 换算成绝对亮度（nit）。
2. **源峰值**：默认按 maxRGB 统计内容峰值（与 MaxCLL 的定义一致，取 99.99 百分位），也可用 `--source-peak` 指定。
3. **EETF**：HDR 参考白（默认 203 nit，BT.2408）对齐到 SDR 100% 白；在 PQ 域中，拐点 `KS = 1.5·maxLum − 0.5` 以下 1:1 保留，以上用 Hermite 样条滚降到目标峰值。
4. **色域**：BT.2020 → sRGB / P3，超出色域的颜色沿亮度轴向灰色收拢（保持亮度与色相）。
5. **编码**：sRGB 传递曲线，8 bit 输出加轻微抖动以减少色带。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--sdr-white` | 203 | 映射为 SDR 白的 HDR 亮度（nit） |
| `--target-peak` | = sdr-white | EETF 目标峰值 Lmax |
| `--target-black` | 0 | 目标黑位 Lmin，>0 时启用暗部 toe |
| `--source-peak` | auto | 源峰值 Lw（nit） |
| `--peak-percentile` | 99.99 | auto 模式的统计百分位 |
| `--knee-offset` | 0.5 | `KS=(1+k)·maxLum−k`，0.5 为 BT.2390 规范值；越小拐点越高 |
| `--mode` | maxrgb | `maxrgb` 保色相 / `luminance` 亮度比例 / `rgb` 逐通道 |
| `--transfer` `--primaries` | 自动 | 覆盖自动识别的传递函数 / 原色 |
| `--hlg-peak` | 1000 | HLG OOTF 假定的显示峰值 |
| `--gamut` | srgb | 输出色域 `srgb` / `p3` |
| `--gamut-map` | compress | `compress` / `clip` |
| `--bits` | 8 | 8 / 16 |
| `--quality` | 95 | JPEG / WebP 质量 |
| `--format` | jpg | 未指定 `-o` 时的输出格式：jpg / png / tif / webp |
| `-j` `--threads` | 0 | 线程数，0 = 全部 CPU 核心 |
| `--png-level` | 6 | PNG 压缩级别 0–9 |
| `--no-optimize` | | JPEG 不做霍夫曼表优化（更快，文件约大 40%） |
| `--no-exif` | | 不保留 EXIF（默认保留，但方向置 1、删除 MakerNote） |

## 性能

4800 万像素（8064×6048）PQ 照片，2 核 CPU：

| 任务 | v1.0 | v1.1 |
|---|---|---|
| AVIF → JPEG | 19.3 s / 3.6 GB | 4.9 s / 0.8 GB |
| AVIF → 16 bit PNG | 27.7 s / 3.6 GB | 8.5 s / 1.2 GB |
| EXR → JPEG | 21.2 s / 3.7 GB | 4.7 s / 1.1 GB |

整数码值全部经查找表转换（无逐像素幂运算）、全程 float32、按 256 行分块多线程处理、色域映射只作用于超色域像素、PNG 分块并行压缩。核心越多越快；剩余时间主要是解码与 JPEG 编码。

## 开发

```bash
pip install -e ".[test]"
pytest
```

修改 `hdr2sdr_eetf.py` 中的 `__version__` 并推送到 `main`，Release 工作流会自动打 `v<版本号>` 标签、构建四个平台的可执行文件并发布；手动推送 `v*` 标签也可以。

## 参考

- ITU-R BT.2390 — High dynamic range television for production and international programme exchange（EETF）
- ITU-R BT.2408 — Operational practices in HDR television production（203 nit 参考白）
- ITU-R BT.2100 — PQ / HLG
