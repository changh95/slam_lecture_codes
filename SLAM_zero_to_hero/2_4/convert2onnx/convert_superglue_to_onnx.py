#!/usr/bin/env python3
"""
Convert SuperGlue PyTorch Model to ONNX

This script exports the SuperGlue model to ONNX format for TensorRT deployment.

Usage:
    python convert_superglue_to_onnx.py --weights superglue_indoor.pth --output superglue_indoor.onnx

Requirements:
    pip install torch onnx onnxsim

Important Notes:
1. SuperGlue has DYNAMIC input shapes (variable number of keypoints)
2. TensorRT requires optimization profiles for dynamic shapes
3. The Sinkhorn iterations may need adjustment for ONNX export

Available pretrained models from Magic Leap:
- superglue_indoor.pth: Trained on ScanNet (indoor scenes)
- superglue_outdoor.pth: Trained on MegaDepth (outdoor/nature scenes)
"""

import argparse
import torch
import torch.onnx
import onnx

from superglue import SuperGlue


def export_to_onnx(model, output_path, num_keypoints_0=512, num_keypoints_1=512, opset_version=11):
    """
    Export SuperGlue PyTorch model to ONNX format.

    Args:
        model: PyTorch SuperGlue model
        output_path: Path to save ONNX model
        num_keypoints_0: Reference number of keypoints for image 0
        num_keypoints_1: Reference number of keypoints for image 1
        opset_version: ONNX opset version (11+ for TensorRT compatibility)
    """
    model.eval()

    # Create dummy inputs
    # Note: We use int32 for dynamic axes compatibility with TensorRT
    keypoints_0 = torch.randn(1, num_keypoints_0, 2)
    scores_0 = torch.rand(1, num_keypoints_0)
    descriptors_0 = torch.randn(1, 256, num_keypoints_0)
    descriptors_0 = torch.nn.functional.normalize(descriptors_0, dim=1)

    keypoints_1 = torch.randn(1, num_keypoints_1, 2)
    scores_1 = torch.rand(1, num_keypoints_1)
    descriptors_1 = torch.randn(1, 256, num_keypoints_1)
    descriptors_1 = torch.nn.functional.normalize(descriptors_1, dim=1)

    dummy_inputs = (keypoints_0, scores_0, descriptors_0,
                    keypoints_1, scores_1, descriptors_1)

    print(f"Exporting to ONNX...")
    print(f"  Reference keypoints: {num_keypoints_0} x {num_keypoints_1}")

    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=[
            'keypoints_0', 'scores_0', 'descriptors_0',
            'keypoints_1', 'scores_1', 'descriptors_1'
        ],
        output_names=['scores'],
        dynamic_axes={
            # Image 0 keypoints: variable N
            'keypoints_0': {0: 'batch', 1: 'num_keypoints_0'},
            'scores_0': {0: 'batch', 1: 'num_keypoints_0'},
            'descriptors_0': {0: 'batch', 2: 'num_keypoints_0'},
            # Image 1 keypoints: variable M
            'keypoints_1': {0: 'batch', 1: 'num_keypoints_1'},
            'scores_1': {0: 'batch', 1: 'num_keypoints_1'},
            'descriptors_1': {0: 'batch', 2: 'num_keypoints_1'},
            # Output: (N+1) x (M+1)
            'scores': {0: 'batch', 1: 'num_keypoints_0_plus_1', 2: 'num_keypoints_1_plus_1'}
        }
    )

    print(f"Saved ONNX model to {output_path}")

    # Verify the model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    return output_path


def simplify_onnx(input_path, output_path, num_kp0=512, num_kp1=512):
    """
    Simplify ONNX model using onnx-simplifier.

    For SuperGlue, we need to provide input shapes for the dynamic dimensions.
    """
    try:
        import onnxsim
    except ImportError:
        print("onnxsim not installed. Skipping simplification.")
        print("Install with: pip install onnxsim")
        return input_path

    print("Simplifying ONNX model...")
    model = onnx.load(input_path)

    # Specify shapes for dynamic axes
    input_shapes = {
        'keypoints_0': [1, num_kp0, 2],
        'scores_0': [1, num_kp0],
        'descriptors_0': [1, 256, num_kp0],
        'keypoints_1': [1, num_kp1, 2],
        'scores_1': [1, num_kp1],
        'descriptors_1': [1, 256, num_kp1],
    }

    try:
        model_simp, check = onnxsim.simplify(
            model,
            dynamic_input_shape=True,
            input_shapes=input_shapes
        )

        if check:
            onnx.save(model_simp, output_path)
            print(f"Saved simplified model to {output_path}")
            return output_path
        else:
            print("Simplification check failed, using original model")
            return input_path
    except Exception as e:
        print(f"Simplification failed: {e}")
        print("Using original model")
        return input_path


def main():
    parser = argparse.ArgumentParser(description='Convert SuperGlue to ONNX')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights (.pth)')
    parser.add_argument('--output', type=str, default='superglue_indoor.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model with onnx-simplifier')
    parser.add_argument('--num-keypoints', type=int, default=512,
                        help='Reference number of keypoints for export')
    parser.add_argument('--sinkhorn-iters', type=int, default=20,
                        help='Number of Sinkhorn iterations (fewer for faster export)')
    args = parser.parse_args()

    # Create model with reduced Sinkhorn iterations for ONNX export
    print("Creating SuperGlue model...")
    model = SuperGlue(sinkhorn_iters=args.sinkhorn_iters)

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
        num_keypoints_0=args.num_keypoints,
        num_keypoints_1=args.num_keypoints
    )

    # Optionally simplify
    if args.simplify:
        output_simp = args.output.replace('.onnx', '_sim.onnx')
        simplify_onnx(onnx_path, output_simp, args.num_keypoints, args.num_keypoints)

    print("\nDone!")
    print("\nImportant: TensorRT needs optimization profiles for dynamic shapes")
    print("The C++ code sets up profiles for 1-1024 keypoints per image")
    print("\nNext steps:")
    print("1. Copy ONNX file to weights/ directory")
    print("2. Update config/config.yaml with correct filename")
    print("3. Run inference (TensorRT engine will be built on first run)")


if __name__ == '__main__':
    main()
