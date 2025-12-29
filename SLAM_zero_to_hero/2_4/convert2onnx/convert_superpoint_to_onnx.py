#!/usr/bin/env python3
"""
Convert SuperPoint PyTorch Model to ONNX

This script exports the SuperPoint model to ONNX format for TensorRT deployment.

Usage:
    python convert_superpoint_to_onnx.py --weights superpoint_v1.pth --output superpoint_v1.onnx

Requirements:
    pip install torch onnx onnxsim

Steps for full deployment:
1. Download pretrained weights from:
   https://github.com/magicleap/SuperGluePretrainedNetwork

2. Run this script to export to ONNX

3. (Optional) Simplify ONNX model for better TensorRT compatibility:
   python -m onnxsim superpoint.onnx superpoint_sim.onnx

4. TensorRT will convert ONNX to optimized engine at runtime
"""

import argparse
import torch
import torch.onnx
import onnx

from superpoint import SuperPoint


def export_to_onnx(model, output_path, input_shape=(1, 1, 480, 640), opset_version=11):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        input_shape: (batch, channels, height, width)
        opset_version: ONNX opset version (11+ recommended for TensorRT)
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Export to ONNX
    print(f"Exporting to ONNX with input shape {input_shape}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['scores', 'descriptors'],
        dynamic_axes={
            'input': {
                0: 'batch_size',
                2: 'height',
                3: 'width'
            },
            'scores': {
                0: 'batch_size',
                1: 'height',
                2: 'width'
            },
            'descriptors': {
                0: 'batch_size',
                2: 'height_div8',
                3: 'width_div8'
            }
        }
    )

    print(f"Saved ONNX model to {output_path}")

    # Verify the model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    return output_path


def simplify_onnx(input_path, output_path):
    """
    Simplify ONNX model using onnx-simplifier.

    This can improve TensorRT conversion by:
    - Folding constants
    - Removing unused nodes
    - Fusing operations
    """
    try:
        import onnxsim
    except ImportError:
        print("onnxsim not installed. Skipping simplification.")
        print("Install with: pip install onnxsim")
        return input_path

    print("Simplifying ONNX model...")
    model = onnx.load(input_path)

    # Simplify
    model_simp, check = onnxsim.simplify(
        model,
        dynamic_input_shape=True,
        input_shapes={'input': [1, 1, 480, 640]}
    )

    if check:
        onnx.save(model_simp, output_path)
        print(f"Saved simplified model to {output_path}")
        return output_path
    else:
        print("Simplification failed, using original model")
        return input_path


def main():
    parser = argparse.ArgumentParser(description='Convert SuperPoint to ONNX')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights (.pth)')
    parser.add_argument('--output', type=str, default='superpoint_v1.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model with onnx-simplifier')
    parser.add_argument('--height', type=int, default=480,
                        help='Reference input height')
    parser.add_argument('--width', type=int, default=640,
                        help='Reference input width')
    args = parser.parse_args()

    # Create model
    print("Creating SuperPoint model...")
    model = SuperPoint()

    # Load pretrained weights if provided
    if args.weights:
        print(f"Loading weights from {args.weights}...")
        state_dict = torch.load(args.weights, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)

    # Export to ONNX
    onnx_path = export_to_onnx(
        model,
        args.output,
        input_shape=(1, 1, args.height, args.width)
    )

    # Optionally simplify
    if args.simplify:
        output_simp = args.output.replace('.onnx', '_sim.onnx')
        simplify_onnx(onnx_path, output_simp)

    print("\nDone!")
    print("\nNext steps:")
    print("1. Copy ONNX file to weights/ directory")
    print("2. Update config/config.yaml with correct filename")
    print("3. Run inference (TensorRT engine will be built on first run)")


if __name__ == '__main__':
    main()
