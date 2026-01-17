#!/usr/bin/env python3
"""
DetectLayoutTRT.py

High-throughput layout detector using:
- TensorRT FP16 engine
- pinned host memory
- batched inference
- overlapped H2D/D2H transfers via CUDA streams

CLI (compatible with TagMyPDF.py):

    python DetectLayoutTRT.py debug_dir layout_out dpi --pdf input.pdf --debugpng \
        --weights yolov8n-doclaynet.engine --labelmap labelmap.json
"""

import os, sys, glob, json, argparse, math, time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import cv2

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)


# ---------- Utilities ----------

def log(msg):
    print(msg, flush=True)


def sorted_page_images(debug_dir: str) -> List[str]:
    # Matches what ProbingFile.py typically writes
    files = sorted(glob.glob(os.path.join(debug_dir, "preview_p*.png")))
    if not files:
        raise FileNotFoundError(f"No preview_p*.png found in {debug_dir}")
    return files


def load_labelmap(path: str) -> Dict[int, str]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # allow either { "0": "Title"} or { 0: "Title" }
    out = {}
    for k, v in data.items():
        out[int(k)] = str(v)
    return out


# ---------- TensorRT wrapper ----------

@dataclass
class TrtIO:
    engine: trt.ICudaEngine
    context: trt.IExecutionContext
    input_binding: int
    output_binding: int
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    d_input: cuda.DeviceAllocation
    d_output: cuda.DeviceAllocation
    h_input: np.ndarray      # pinned host
    h_output: np.ndarray     # pinned host
    stream: cuda.Stream


def load_trt_engine(path: str, max_batch: int) -> TrtIO:
    if not os.path.exists(path):
        raise FileNotFoundError(f"TensorRT engine not found: {path}")

    logger = trt.Logger(trt.Logger.INFO)
    with open(path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")

    context = engine.create_execution_context()

    # Assume a single input + single output
    assert engine.num_bindings == 2, "This sample assumes 1 input & 1 output binding."

    input_binding = 0 if engine.binding_is_input(0) else 1
    output_binding = 1 - input_binding

    input_name = engine.get_binding_name(input_binding)
    output_name = engine.get_binding_name(output_binding)
    input_shape = engine.get_binding_shape(input_binding)  # e.g. (B,3,640,640)
    output_shape = engine.get_binding_shape(output_binding)  # e.g. (B, num_boxes, 6)

    # Make the batch dimension dynamic if engine supports it
    if input_shape[0] == -1:
        # set an optimization batch size
        shape = (max_batch, *input_shape[1:])
        context.set_binding_shape(input_binding, shape)
        input_shape = shape
    else:
        max_batch = input_shape[0]

    if output_shape[0] == -1:
        out_shape = (max_batch, *output_shape[1:])
        context.set_binding_shape(output_binding, out_shape)
        output_shape = out_shape

    # Allocate pinned host buffers
    h_input = cuda.pagelocked_empty(
        trt.volume(input_shape),
        dtype=np.float32
    )
    h_output = cuda.pagelocked_empty(
        trt.volume(output_shape),
        dtype=np.float32
    )

    # Allocate device buffers
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    log(f"Loaded TensorRT engine '{os.path.basename(path)}'")
    log(f"  input ({input_name}): shape={input_shape}")
    log(f"  output({output_name}): shape={output_shape}")

    return TrtIO(
        engine=engine,
        context=context,
        input_binding=input_binding,
        output_binding=output_binding,
        input_shape=input_shape,
        output_shape=output_shape,
        d_input=d_input,
        d_output=d_output,
        h_input=h_input,
        h_output=h_output,
        stream=stream,
    )


# ---------- Pre/post-processing ----------

def letterbox(img: np.ndarray, new_shape=(1280, 1280)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize + pad image to letterbox shape, keeping aspect ratio.
    Returns resized image, scale, padding (dw, dh).
    """
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_shape[1], new_shape[0], 3), 114, dtype=np.uint8)
    dw = (new_shape[0] - nw) // 2
    dh = (new_shape[1] - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw, :] = resized
    return canvas, r, (dw, dh)


def preprocess_batch(
    img_paths: List[str],
    io: TrtIO
) -> Tuple[np.ndarray, List[Tuple[float, Tuple[int, int], Tuple[int, int]]]]:
    """
    Load images, letterbox to engine size, normalize to [0,1], CHW, batched.
    Returns:
      batch array (float32) with shape input_shape
      meta list: (scale, (dw,dh), (orig_w, orig_h)) per image
    """
    b, c, h, w = io.input_shape
    batch = np.zeros((b, c, h, w), dtype=np.float32)
    meta = []

    for i, p in enumerate(img_paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image {p}")
        orig_h, orig_w = img.shape[:2]
        lb, scale, (dw, dh) = letterbox(img, new_shape=(w, h))
        # BGR -> RGB, HWC -> CHW, normalize 0..1
        lb = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        batch[i] = lb
        meta.append((scale, (dw, dh), (orig_w, orig_h)))

    return batch, meta


def postprocess_trt_output(
    outputs: np.ndarray,
    meta: List[Tuple[float, Tuple[int, int], Tuple[int, int]]],
    conf_thres: float = 0.25
) -> List[List[Dict]]:
    """
    Convert TensorRT outputs to per-image detection dicts.

    Expected output shape: (B, N, 6) with [x1,y1,x2,y2,conf,cls] in letterboxed coords.
    Returns list of detections per image:
        [{'bbox':[x0,y0,x1,y1], 'conf':0.91, 'cls':int}, ...]
    """
    b = outputs.shape[0]
    all_dets = []

    for i in range(b):
        scale, (dw, dh), (orig_w, orig_h) = meta[i]
        dets = []
        for row in outputs[i]:
            x1, y1, x2, y2, conf, cls_id = row.tolist()
            if conf < conf_thres:
                continue

            # Undo letterbox to get original image coords
            x1 -= dw
            x2 -= dw
            y1 -= dh
            y2 -= dh
            x1 /= scale
            x2 /= scale
            y1 /= scale
            y2 /= scale

            # clamp
            x1 = max(0, min(orig_w - 1, x1))
            x2 = max(0, min(orig_w - 1, x2))
            y1 = max(0, min(orig_h - 1, y1))
            y2 = max(0, min(orig_h - 1, y2))

            dets.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf),
                "cls": int(cls_id)
            })
        all_dets.append(dets)
    return all_dets


# ---------- Inference ----------

def infer_batch(io: TrtIO, batch: np.ndarray) -> np.ndarray:
    """
    Copy a batch to device (pinned host -> device) and run async inference.
    Returns outputs as np.ndarray with shape io.output_shape (but batch dimension = len(batch)).
    """
    # flatten into pinned buffer
    np.copyto(io.h_input, batch.ravel())

    # H2D async
    cuda.memcpy_htod_async(io.d_input, io.h_input, io.stream)

    # Set bindings
    bindings = [int(io.d_input), int(io.d_output)]
    io.context.execute_async_v2(bindings=bindings, stream_handle=io.stream.handle)

    # D2H async
    cuda.memcpy_dtoh_async(io.h_output, io.d_output, io.stream)

    # wait
    io.stream.synchronize()

    # reshape to output_shape
    out = np.array(io.h_output).reshape(io.output_shape)
    return out


# ---------- Main layout detection ----------

def detect_layout_trt(
    debug_dir: str,
    layout_out: str,
    dpi: int,
    engine_path: str,
    labelmap_path: str,
    batch_size: int = 8,
    conf_thres: float = 0.25
):
    os.makedirs(layout_out, exist_ok=True)
    pages = sorted_page_images(debug_dir)
    labelmap = load_labelmap(labelmap_path)

    io = load_trt_engine(engine_path, max_batch=batch_size)
    B = io.input_shape[0]

    num_pages = len(pages)
    log(f"Found {num_pages} pages for layout detection")

    t0 = time.perf_counter()
    processed = 0

    # Process in batches
    for start in range(0, num_pages, B):
        end = min(start + B, num_pages)
        batch_paths = pages[start:end]
        cur_bs = len(batch_paths)

        # Preprocess on CPU (fast enough; could be moved to GPU later)
        batch, meta = preprocess_batch(batch_paths, io)

        # Run inference
        outputs = infer_batch(io, batch)

        # Only keep first cur_bs if last batch is smaller
        outputs = outputs[:cur_bs]

        # Post-process
        dets_per_img = postprocess_trt_output(outputs, meta, conf_thres=conf_thres)

        # Write layout JSON per page
        for idx, dets in enumerate(dets_per_img):
            page_idx = start + idx + 1  # pages are 1-based
            layout = {
                "page": page_idx,
                "dpi": dpi,
                "elements": []
            }
            for d in dets:
                cls_id = d["cls"]
                label = labelmap.get(cls_id, str(cls_id))
                x0, y0, x1, y1 = d["bbox"]
                layout["elements"].append({
                    "label": label,
                    "bbox": [x0, y0, x1, y1],
                    "score": d["conf"],
                    "class_id": cls_id
                })

            out_path = os.path.join(layout_out, f"page_{page_idx:03d}.layout.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(layout, f, ensure_ascii=False, indent=2)

            processed += 1

    dt = time.perf_counter() - t0
    log(f"Processed {processed} pages in {dt:.3f}s  ({processed / dt:.2f} pages/sec)")


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(description="High-throughput layout detector (TensorRT)")
    parser.add_argument("debug_dir", help="Directory containing preview_p*.png from ProbingFile.py")
    parser.add_argument("layout_out", help="Output directory for page_XXX.layout.json")
    parser.add_argument("dpi", help="DPI used when rendering pages (for metadata only)")
    parser.add_argument("--pdf", required=True, help="Original PDF path (not used directly, kept for CLI compat)")
    parser.add_argument("--debugpng", action="store_true", help="Ignored, kept for CLI compat")
    parser.add_argument("--weights", required=True, help="TensorRT engine path (.engine)")
    parser.add_argument("--labelmap", default="", help="Label map JSON path")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    detect_layout_trt(
        debug_dir=args.debug_dir,
        layout_out=args.layout_out,
        dpi=int(args.dpi),
        engine_path=args.weights,
        labelmap_path=args.labelmap,
        batch_size=args.batch,
        conf_thres=args.conf
    )


if __name__ == "__main__":
    main()
