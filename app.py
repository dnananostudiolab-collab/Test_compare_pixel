
import io
import struct
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from rapidfuzz.distance import Levenshtein
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="PNG One-Pixel File Inspector",
    page_icon="🧪",
    layout="wide",
)

st.title("PNG One-Pixel File Inspector")
st.caption(
    "Upload one image, convert it to Grayscale or RGB, change exactly one pixel "
    "in a copy, save both images as PNG with identical encoder settings, "
    "and compare the actual PNG file contents."
)


# ============================================================
# HELPERS
# ============================================================

def encode_png(arr, color_mode, compress_level=6, optimize=False):
    """
    Encode BOTH original and modified images through this exact same function.
    """
    if color_mode == "Grayscale":
        arr = arr.astype(np.uint8)
        pil = Image.fromarray(arr)
    else:
        arr = arr.astype(np.uint8)
        pil = Image.fromarray(arr, "RGB")

    buffer = io.BytesIO()
    pil.save(
        buffer,
        format="PNG",
        compress_level=int(compress_level),
        optimize=bool(optimize),
    )
    return buffer.getvalue()


def decode_png(data, color_mode):
    mode = "L" if color_mode == "Grayscale" else "RGB"
    return np.asarray(
        Image.open(io.BytesIO(data)).convert(mode),
        dtype=np.uint8
    )


def printable_ascii(byte_value):
    return chr(byte_value) if 32 <= byte_value <= 126 else "."


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def fixed_offset_difference(a, b):
    common = min(len(a), len(b))
    changed = []
    changed_bits = 0

    for i in range(common):
        if a[i] != b[i]:
            bits = (a[i] ^ b[i]).bit_count()
            changed.append((i, a[i], b[i], bits))
            changed_bits += bits

    return changed, changed_bits, abs(len(a) - len(b))


def sequence_metrics(a, b):
    if not RAPIDFUZZ_AVAILABLE:
        return None

    ops = Levenshtein.editops(a, b)

    substitutions = 0
    insertions = 0
    deletions = 0
    bit_burden = 0

    for op in ops:
        if op.tag == "replace":
            substitutions += 1
            bit_burden += (a[op.src_pos] ^ b[op.dest_pos]).bit_count()
        elif op.tag == "insert":
            insertions += 1
            bit_burden += 8
        elif op.tag == "delete":
            deletions += 1
            bit_burden += 8

    edit_distance = substitutions + insertions + deletions
    denom = max(len(a), len(b), 1)

    return {
        "edit_distance": edit_distance,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "edit_rate": edit_distance / denom,
        "binary_edit_burden": bit_burden / (8 * denom),
    }


def byte_diff_dataframe(a, b, max_rows=None):
    common = min(len(a), len(b))
    rows = []

    for i in range(common):
        x = a[i]
        y = b[i]

        if x != y:
            rows.append({
                "Offset (dec)": i,
                "Offset (hex)": f"0x{i:08X}",
                "Original dec": x,
                "Modified dec": y,
                "Original hex": f"{x:02X}",
                "Modified hex": f"{y:02X}",
                "Original binary": f"{x:08b}",
                "Modified binary": f"{y:08b}",
                "Different bits": (x ^ y).bit_count(),
                "Original ASCII": printable_ascii(x),
                "Modified ASCII": printable_ascii(y),
                "ASCII change": f"{printable_ascii(x)} → {printable_ascii(y)}",
            })

            if max_rows is not None and len(rows) >= max_rows:
                break

    return pd.DataFrame(rows)


def hex_ascii_dump(data, bytes_per_line=16):
    lines = []

    for offset in range(0, len(data), bytes_per_line):
        chunk = data[offset:offset + bytes_per_line]

        hex_part = " ".join(f"{b:02X}" for b in chunk)
        ascii_part = "".join(printable_ascii(b) for b in chunk)

        lines.append(
            f"{offset:08X}  {hex_part:<47}  |{ascii_part}|"
        )

    return "\n".join(lines)


def binary_dump(data, bytes_per_line=8):
    lines = []

    for offset in range(0, len(data), bytes_per_line):
        chunk = data[offset:offset + bytes_per_line]
        bits = " ".join(f"{b:08b}" for b in chunk)
        lines.append(f"{offset:08X}  {bits}")

    return "\n".join(lines)


def parse_png_chunks(data):
    signature = b"\x89PNG\r\n\x1a\n"
    if data[:8] != signature:
        raise ValueError("Invalid PNG signature.")

    rows = []
    pos = 8

    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos:pos+4])[0]
        ctype = data[pos+4:pos+8].decode("ascii", errors="replace")
        data_start = pos + 8
        data_end = data_start + length
        crc_start = data_end
        end = crc_start + 4

        rows.append({
            "Chunk": ctype,
            "Start offset": pos,
            "Data length": length,
            "End offset": end - 1,
        })

        pos = end

        if ctype == "IEND":
            break

    return pd.DataFrame(rows)


def nearest_preview(arr, color_mode, scale=10):
    if color_mode == "Grayscale":
        pil = Image.fromarray(arr)
    else:
        pil = Image.fromarray(arr, "RGB")

    return pil.resize(
        (pil.width * scale, pil.height * scale),
        Image.Resampling.NEAREST
    )


def value_text(value, mode):
    if mode == "Grayscale":
        return str(int(value))

    return (
        f"R={int(value[0])}, "
        f"G={int(value[1])}, "
        f"B={int(value[2])}"
    )


# ============================================================
# 1. UPLOAD
# ============================================================

st.subheader("1. Upload image")

uploaded = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if uploaded is None:
    st.info("Upload an image to start the experiment.")
    st.stop()


# ============================================================
# 2. PROCESSING MODE
# ============================================================

st.subheader("2. Choose image representation")

mode_col, info_col = st.columns([1, 1])

with mode_col:
    color_mode = st.radio(
        "Working color mode",
        ["Grayscale", "RGB"],
        horizontal=True,
    )

with info_col:
    st.caption(
        "The uploaded file is decoded once, then converted to the selected working mode. "
        "Both experimental PNGs are created from this same processed image."
    )

source_pil = Image.open(uploaded)

if color_mode == "Grayscale":
    processed_pil = source_pil.convert("L")
    original = np.asarray(processed_pil, dtype=np.uint8)
else:
    processed_pil = source_pil.convert("RGB")
    original = np.asarray(processed_pil, dtype=np.uint8)

height, width = original.shape[:2]

i1, i2, i3, i4 = st.columns(4)

with i1:
    st.metric("Width", f"{width} px")

with i2:
    st.metric("Height", f"{height} px")

with i3:
    st.metric("Mode", color_mode)

with i4:
    st.metric("dtype", str(original.dtype))


# ============================================================
# 3. SELECT AND MODIFY ONE PIXEL
# ============================================================

st.subheader("3. Create one-pixel error")

c1, c2 = st.columns(2)

with c1:
    row = st.number_input(
        "Pixel row",
        min_value=0,
        max_value=height - 1,
        value=0,
        step=1,
    )

with c2:
    col = st.number_input(
        "Pixel column",
        min_value=0,
        max_value=width - 1,
        value=0,
        step=1,
    )

row = int(row)
col = int(col)

modified = original.copy()

if color_mode == "Grayscale":
    original_value = int(original[row, col])

    st.write(f"Original pixel value: **{original_value}**")

    new_value = st.slider(
        "New grayscale value",
        min_value=0,
        max_value=255,
        value=255 if original_value != 255 else 0,
        step=1,
    )

    modified[row, col] = int(new_value)

else:
    original_value = original[row, col].copy()

    st.write(
        "Original RGB value: "
        f"**R={int(original_value[0])}, "
        f"G={int(original_value[1])}, "
        f"B={int(original_value[2])}**"
    )

    r1, r2, r3 = st.columns(3)

    with r1:
        new_r = st.slider(
            "New R",
            0, 255,
            255 if int(original_value[0]) != 255 else 0
        )

    with r2:
        new_g = st.slider(
            "New G",
            0, 255,
            255 if int(original_value[1]) != 255 else 0
        )

    with r3:
        new_b = st.slider(
            "New B",
            0, 255,
            255 if int(original_value[2]) != 255 else 0
        )

    modified[row, col] = [
        int(new_r),
        int(new_g),
        int(new_b),
    ]

# Pixel validation
if color_mode == "Grayscale":
    changed_pixel_mask = original != modified
    changed_pixel_count = int(np.sum(changed_pixel_mask))
else:
    changed_pixel_mask = np.any(original != modified, axis=2)
    changed_pixel_count = int(np.sum(changed_pixel_mask))

if changed_pixel_count == 0:
    st.warning("The new pixel value is identical to the original value.")
    st.stop()

if changed_pixel_count != 1:
    st.error("Validation failed: more than one pixel changed.")
    st.stop()

st.success(
    f"Validated: exactly one pixel changed at [{row}, {col}]."
)


# ============================================================
# 4. IMAGE PAIR
# ============================================================

st.subheader("4. Image pair")

left_img, right_img = st.columns(2)

with left_img:
    st.markdown("#### Original processed image")
    st.image(
        original,
        caption=f"Pixel [{row}, {col}] = {value_text(original[row, col], color_mode)}",
        width="stretch",
    )

with right_img:
    st.markdown("#### Modified image")
    st.image(
        modified,
        caption=f"Pixel [{row}, {col}] = {value_text(modified[row, col], color_mode)}",
        width="stretch",
    )


# ============================================================
# 5. PNG SAVE SETTINGS
# ============================================================

st.subheader("5. Save both images as PNG")

s1, s2 = st.columns(2)

with s1:
    compress_level = st.slider(
        "PNG compression level",
        0, 9, 6
    )

with s2:
    optimize = st.checkbox(
        "Optimize",
        value=False,
    )

original_png = encode_png(
    original,
    color_mode,
    compress_level,
    optimize,
)

modified_png = encode_png(
    modified,
    color_mode,
    compress_level,
    optimize,
)

# Strict lossless verification
decoded_original = decode_png(original_png, color_mode)
decoded_modified = decode_png(modified_png, color_mode)

assert np.array_equal(decoded_original, original)
assert np.array_equal(decoded_modified, modified)

if color_mode == "Grayscale":
    decoded_changed_pixels = int(
        np.sum(decoded_original != decoded_modified)
    )
else:
    decoded_changed_pixels = int(
        np.sum(np.any(decoded_original != decoded_modified, axis=2))
    )

assert decoded_changed_pixels == 1

st.caption(
    "Both files use the same PNG encoder function, color mode, dimensions, "
    "compression level, optimize setting, and runtime. The only image-data difference "
    "is the selected pixel."
)

d1, d2 = st.columns(2)

with d1:
    st.download_button(
        "Download original.png",
        data=original_png,
        file_name="original.png",
        mime="image/png",
        use_container_width=True,
    )

with d2:
    st.download_button(
        "Download modified_one_pixel.png",
        data=modified_png,
        file_name="modified_one_pixel.png",
        mime="image/png",
        use_container_width=True,
    )


# ============================================================
# 6. ACTUAL FILE SUMMARY
# ============================================================

st.subheader("6. Compare the actual PNG files")

fixed_diffs, fixed_bit_diffs, extra_bytes = fixed_offset_difference(
    original_png,
    modified_png
)

seq = sequence_metrics(
    original_png,
    modified_png
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Original PNG",
        f"{len(original_png)} bytes"
    )

with m2:
    st.metric(
        "Modified PNG",
        f"{len(modified_png)} bytes",
        delta=f"{len(modified_png)-len(original_png):+d} bytes"
    )

with m3:
    st.metric(
        "Changed pixel",
        "1"
    )

with m4:
    if seq is not None:
        st.metric(
            "Aligned file edit rate",
            f'{seq["edit_rate"]*100:.2f}%'
        )
    else:
        st.metric(
            "Different byte offsets",
            len(fixed_diffs) + extra_bytes
        )

if len(original_png) == len(modified_png):
    st.success(
        "The two PNG files have the same file length, but their contents can still differ."
    )
else:
    st.warning(
        "The two PNG files have different lengths. Direct same-offset comparison can "
        "overstate differences after an insertion/deletion, so the alignment-aware "
        "metrics are shown as the primary comparison."
    )


# ============================================================
# 7. FILE CONTENT TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Changed bytes",
        "Hex + ASCII file view",
        "Binary file view",
        "PNG structure",
    ]
)


# ------------------------------------------------------------
# TAB 1 — CHANGED BYTES
# ------------------------------------------------------------

with tab1:

    st.markdown("### Byte-level differences")

    if seq is not None:
        stats = pd.DataFrame([
            ["File edit distance", seq["edit_distance"]],
            ["Substitutions", seq["substitutions"]],
            ["Insertions", seq["insertions"]],
            ["Deletions", seq["deletions"]],
            ["Normalized edit rate", f'{seq["edit_rate"]*100:.3f}%'],
            ["Binary edit burden", f'{seq["binary_edit_burden"]*100:.3f}%'],
        ], columns=["Metric", "Value"])

        st.dataframe(
            stats,
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "The table below compares bytes at the same file offset. "
        "For different-length files, use it mainly to inspect individual byte values; "
        "the alignment-aware metrics above are more reliable for total difference."
    )

    limit = st.selectbox(
        "Rows to display",
        [50, 100, 250, 500, "All"],
        index=1,
    )

    max_rows = None if limit == "All" else int(limit)

    diff_df = byte_diff_dataframe(
        original_png,
        modified_png,
        max_rows=max_rows,
    )

    if len(diff_df):
        st.dataframe(
            diff_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No changed bytes at common offsets.")

    diff_csv = byte_diff_dataframe(
        original_png,
        modified_png,
        max_rows=None,
    ).to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download byte_difference.csv",
        data=diff_csv,
        file_name="byte_difference.csv",
        mime="text/csv",
    )

    st.info(
        "ASCII is only a readable view of byte values. "
        "For example, byte 0x63 can be displayed as 'c'. "
        "PNG itself is a binary file, not a text file."
    )


# ------------------------------------------------------------
# TAB 2 — HEX + ASCII
# ------------------------------------------------------------

with tab2:

    st.markdown("### Actual PNG byte content")

    hex_original = hex_ascii_dump(original_png)
    hex_modified = hex_ascii_dump(modified_png)

    h1, h2 = st.columns(2)

    with h1:
        st.markdown("#### original.png")
        st.code(hex_original, language=None)

    with h2:
        st.markdown("#### modified_one_pixel.png")
        st.code(hex_modified, language=None)

    hd1, hd2 = st.columns(2)

    with hd1:
        st.download_button(
            "Download original_hex_ascii.txt",
            data=hex_original,
            file_name="original_hex_ascii.txt",
            mime="text/plain",
        )

    with hd2:
        st.download_button(
            "Download modified_hex_ascii.txt",
            data=hex_modified,
            file_name="modified_hex_ascii.txt",
            mime="text/plain",
        )


# ------------------------------------------------------------
# TAB 3 — BINARY
# ------------------------------------------------------------

with tab3:

    st.markdown("### Actual file bytes shown as binary")

    binary_original = binary_dump(original_png)
    binary_modified = binary_dump(modified_png)

    b1, b2 = st.columns(2)

    with b1:
        st.markdown("#### original.png")
        st.code(binary_original, language=None)

    with b2:
        st.markdown("#### modified_one_pixel.png")
        st.code(binary_modified, language=None)

    bd1, bd2 = st.columns(2)

    with bd1:
        st.download_button(
            "Download original_binary.txt",
            data=binary_original,
            file_name="original_binary.txt",
            mime="text/plain",
        )

    with bd2:
        st.download_button(
            "Download modified_binary.txt",
            data=binary_modified,
            file_name="modified_binary.txt",
            mime="text/plain",
        )


# ------------------------------------------------------------
# TAB 4 — PNG STRUCTURE
# ------------------------------------------------------------

with tab4:

    p1, p2 = st.columns(2)

    with p1:
        st.markdown("#### original.png chunks")
        st.dataframe(
            parse_png_chunks(original_png),
            hide_index=True,
            use_container_width=True,
        )

    with p2:
        st.markdown("#### modified_one_pixel.png chunks")
        st.dataframe(
            parse_png_chunks(modified_png),
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# 8. VALIDATION
# ============================================================

with st.expander("Validation / reproducibility"):

    st.json({
        "working_mode": color_mode,
        "image_width": width,
        "image_height": height,
        "pixel_coordinate": [row, col],
        "original_pixel": value_text(original[row, col], color_mode),
        "modified_pixel": value_text(modified[row, col], color_mode),
        "changed_decoded_pixels": decoded_changed_pixels,
        "png_compression_level": int(compress_level),
        "png_optimize": bool(optimize),
        "original_png_bytes": len(original_png),
        "modified_png_bytes": len(modified_png),
        "original_sha256": sha256_bytes(original_png),
        "modified_sha256": sha256_bytes(modified_png),
    })

    st.success(
        "Lossless validation passed: both saved PNGs decode exactly to their source arrays, "
        "and the decoded images differ at exactly one pixel."
    )
