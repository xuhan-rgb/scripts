import argparse
import os
from pathlib import Path
import time

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent


def preprocess_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 352))
    image = image.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(image, dtype=np.float32) / np.float32(255.0)


def get_depth_map(outputs):
    if not outputs:
        raise ValueError("Depth model returned no output tensors.")

    depth_output = outputs[0]
    if depth_output.ndim != 4 or depth_output.shape[0] < 1 or depth_output.shape[1] < 1:
        raise ValueError(
            "Expected the first depth output to have NCHW shape with at least one "
            f"batch and channel, got {depth_output.shape}."
        )
    return depth_output[0, 0]


def colorize_depth(depth_map):
    finite_depth = depth_map[np.isfinite(depth_map)]
    if finite_depth.size == 0:
        raise ValueError("Depth output contains no finite values.")

    vmin = float(finite_depth.min())
    vmax = float(np.percentile(finite_depth, 95))
    if vmax <= vmin:
        vmax = vmin + 1.0

    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
    colorized_image = (mapper.to_rgba(depth_map)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR)


def write_image(output_path, image):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write depth visualization: {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a fixed-shape TensorRT depth model and write a colorized depth map."
    )
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "model" / "depth_model.onnx"
    )
    parser.add_argument(
        "--image", type=Path, default=PROJECT_ROOT / "images" / "depth_image.jpg"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "images" / "depth_colormapped.jpg",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the generated image in a GUI window."
    )
    args = parser.parse_args(argv)

    if args.show and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        parser.error("--show requires a graphical DISPLAY or WAYLAND_DISPLAY session")
    if not args.model.is_file():
        parser.error(f"model file does not exist: {args.model}")

    input_data = preprocess_image(args.image)
    from tensorrt_inference import TensorRTModel

    tensorrt_model = None
    try:
        tensorrt_model = TensorRTModel(onnx_model_path=str(args.model))
        start_time = time.perf_counter()
        outputs = tensorrt_model.infer(input_data)
        print("Inference time:", time.perf_counter() - start_time)
        for output in outputs:
            print(output.shape)
    finally:
        if tensorrt_model is not None:
            tensorrt_model.release_resources()

    colorized_image = colorize_depth(get_depth_map(outputs))
    write_image(args.output, colorized_image)
    print(f"Depth visualization saved to {args.output}")

    if args.show:
        cv2.imshow("depth_colormapped", colorized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
