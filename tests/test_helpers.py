import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import test_onnx_env
from convert_trt import colorize_depth, get_depth_map, preprocess_image, write_image
from get_onnx_dependencies import find_ldd_dependency


class DependencyParsingTests(unittest.TestCase):
    def test_finds_resolved_library_path(self):
        ldd_output = "libcudart.so.11.0 => /opt/cuda/libcudart.so.11.3 (0x0)\n"
        self.assertEqual(
            find_ldd_dependency(ldd_output, "libcudart.so"),
            "/opt/cuda/libcudart.so.11.3",
        )

    def test_returns_none_when_dependency_is_absent(self):
        self.assertIsNone(find_ldd_dependency("libm.so.6 => /lib/libm.so.6\n", "libcudnn.so"))


class OnnxEnvironmentTests(unittest.TestCase):
    def test_cuda_provider_failure_does_not_create_a_model(self):
        with mock.patch.object(
            test_onnx_env,
            "get_cuda_onnxruntime",
            side_effect=RuntimeError("CUDA provider is unavailable"),
        ), mock.patch.object(test_onnx_env, "create_onnx_model") as create_model:
            self.assertEqual(test_onnx_env.main([]), 1)
        create_model.assert_not_called()


class DepthExampleTests(unittest.TestCase):
    def test_preprocess_image_returns_contiguous_float32_nchw(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "input.png"
            image = np.full((10, 20, 3), 128, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            result = preprocess_image(image_path)

        self.assertEqual(result.shape, (1, 3, 352, 640))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_depth_output_shape_is_validated(self):
        valid_output = np.zeros((1, 1, 2, 3), dtype=np.float32)
        self.assertEqual(get_depth_map([valid_output]).shape, (2, 3))

        with self.assertRaisesRegex(ValueError, "NCHW"):
            get_depth_map([np.zeros((1, 2, 3), dtype=np.float32)])

    def test_colorized_depth_is_written(self):
        depth_map = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        colorized_image = colorize_depth(depth_map)
        self.assertEqual(colorized_image.shape, (2, 2, 3))
        self.assertEqual(colorized_image.dtype, np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "depth.jpg"
            write_image(output_path, colorized_image)
            self.assertTrue(output_path.is_file())
            self.assertIsNotNone(cv2.imread(str(output_path)))


if __name__ == "__main__":
    unittest.main()
