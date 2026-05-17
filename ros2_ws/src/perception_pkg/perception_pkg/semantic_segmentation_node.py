import sys
import os
import gc
import importlib
from contextlib import nullcontext
from pathlib import Path
import cv2
import numpy as np
import rclpy
import torch
import time
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy


def _detect_default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class SemanticSegmentationNode(Node):
    def __init__(self):
        super().__init__('semantic_segmentation_node')
        self.bridge = CvBridge()

        self.declare_parameter('input_topic', '/image_raw')
        self.declare_parameter('output_seg_topic', '/segmentation/labels')
        self.declare_parameter('output_prob_topic', '/segmentation/probabilities')
        self.declare_parameter('output_prob_class_topic', '/segmentation/probability_classes')
        self.declare_parameter('output_overlay_topic', '/segmentation/overlay')
        self.declare_parameter('publish_segmentation', False) 
        self.declare_parameter('publish_probabilities', True)
        self.declare_parameter('publish_overlay', False)
        self.declare_parameter('gscnn_root', '/root/existing_models/GSCNN')
        self.declare_parameter('checkpoint_path', '/root/finetuned_models/gscnn_finetuned_mixed_best_final.pth')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('load_model_fp16_on_cuda', False)
        self.declare_parameter('use_channels_last', False)
        self.declare_parameter('resize_width', 640)   # model trained on 640×400 (landscape)
        self.declare_parameter('resize_height', 400)
        self.declare_parameter('max_inference_hz', 0.0)
        self.declare_parameter('probability_threshold', 0.05)
        self.declare_parameter('max_probability_classes', 4)

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_seg_topic = self.get_parameter('output_seg_topic').get_parameter_value().string_value
        self.output_prob_topic = self.get_parameter('output_prob_topic').get_parameter_value().string_value
        self.output_prob_class_topic = self.get_parameter('output_prob_class_topic').get_parameter_value().string_value
        self.output_overlay_topic = self.get_parameter('output_overlay_topic').get_parameter_value().string_value
        self.publish_segmentation = self.get_parameter('publish_segmentation').get_parameter_value().bool_value
        self.publish_probabilities = self.get_parameter('publish_probabilities').get_parameter_value().bool_value
        self.publish_overlay = self.get_parameter('publish_overlay').get_parameter_value().bool_value
        self.gscnn_root = Path(self.get_parameter('gscnn_root').get_parameter_value().string_value)
        self.checkpoint_path = Path(self.get_parameter('checkpoint_path').get_parameter_value().string_value)
        self.use_fp16 = self.get_parameter('use_fp16').get_parameter_value().bool_value
        self.load_model_fp16_on_cuda = self.get_parameter(
            'load_model_fp16_on_cuda'
        ).get_parameter_value().bool_value
        self.use_channels_last = self.get_parameter('use_channels_last').get_parameter_value().bool_value
        self.resize_width = self.get_parameter('resize_width').get_parameter_value().integer_value
        self.resize_height = self.get_parameter('resize_height').get_parameter_value().integer_value
        self.max_inference_hz = self.get_parameter('max_inference_hz').get_parameter_value().double_value
        self.probability_threshold = max(
            0.0,
            float(self.get_parameter('probability_threshold').get_parameter_value().double_value),
        )
        self.max_probability_classes = max(
            1,
            int(self.get_parameter('max_probability_classes').get_parameter_value().integer_value),
        )
        self.min_inference_period = 0.0 if self.max_inference_hz <= 0.0 else 1.0 / self.max_inference_hz

        sensor_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        

        requested_device = self.get_parameter('device').get_parameter_value().string_value
        if requested_device == 'auto':
            requested_device = _detect_default_device()
        self.device = torch.device(requested_device)

        self.class_colors_rgb = np.array([
            [108, 64, 20], # Dirt
            [0, 102, 0], # Grass
            [0, 255, 0], # Tree
            [0, 153, 153], # Pole
            [0, 128, 255], # water
            [0, 0, 255], # Sky
            [255, 255, 0], # Vehicle
            [255, 0, 127], # Object
            [64, 64, 64], # Asphalt
            [255, 0, 0], # Building
            [102, 0, 0], # Log
            [204, 153, 255], # Person
            [102, 0, 204], # Fence
            [255, 153, 204], # Bush
            [170, 170, 170], # Concrete
            [41, 121, 255], # Barrier
            [134, 255, 239], # Puddle
            [99, 66, 34], # Mud
            [110, 22, 138], # Rubble
        ], dtype=np.uint8)

        self.model = self._load_model()
        self.model_dtype = next(self.model.parameters()).dtype
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=self.device,
            dtype=self.model_dtype,
        ).view(3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=self.device,
            dtype=self.model_dtype,
        ).view(3, 1, 1)

        self.segmentation_pub = None
        if self.publish_segmentation:
            self.segmentation_pub = self.create_publisher(
                Image,
                self.output_seg_topic,
                qos_profile=sensor_qos,
            )

        self.probabilities_pub = None
        self.probability_classes_pub = None
        if self.publish_probabilities:
            self.probabilities_pub = self.create_publisher(
                Image,
                self.output_prob_topic,
                qos_profile=sensor_qos,
            )
            self.probability_classes_pub = self.create_publisher(
                Image,
                self.output_prob_class_topic,
                qos_profile=sensor_qos,
            )
        
        self.overlay_pub = None
        if self.publish_overlay:
            self.overlay_pub = self.create_publisher(
                Image, 
                self.output_overlay_topic, 
                qos_profile=sensor_qos)

        self.image_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            qos_profile=sensor_qos
        )

        # Frame counter for periodic memory cleanup
        self.frame_count = 0
        self.cleanup_interval = 100  # Run cleanup every N frames
        self.last_inference_time = 0.0

        self.get_logger().info(
            f'Semantic segmentation node started. input={self.input_topic}, '
            f'segmentation_out={self.output_seg_topic if self.publish_segmentation else "disabled"}, '
            f'overlay_out={self.output_overlay_topic}, '
            f'probabilities_out={self.output_prob_topic if self.publish_probabilities else "disabled"}, '
            f'probability_classes_out={self.output_prob_class_topic if self.publish_probabilities else "disabled"}, '
            f'device={self.device}, max_inference_hz={self.max_inference_hz:.2f}, '
            f'load_model_fp16_on_cuda={self.load_model_fp16_on_cuda}, '
            f'use_channels_last={self.use_channels_last}, '
            f'probability_threshold={self.probability_threshold:.3f}, '
            f'max_probability_classes={self.max_probability_classes}'
        )

    def _load_model(self):
        if not self.gscnn_root.exists():
            raise FileNotFoundError(f'GSCNN root not found: {self.gscnn_root}')
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {self.checkpoint_path}')

        if str(self.gscnn_root) not in sys.path:
            sys.path.insert(0, str(self.gscnn_root))

        gscnn_module = importlib.import_module('network.gscnn')
        GSCNN = gscnn_module.GSCNN

        # Log GPU memory before loading (if using CUDA)
        if self.device.type == 'cuda':
            gc.collect()
            self._safe_empty_cuda_cache()
            self._log_cuda_memory('before model load')

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        model = self._build_model_from_checkpoint(checkpoint, GSCNN)

        try:
            if self.device.type == 'cuda':
                gc.collect()
                self._safe_empty_cuda_cache()
                if self.load_model_fp16_on_cuda:
                    model = model.half()
                    self.get_logger().info('Loading CUDA model weights in float16 to reduce GPU memory usage')
            model = model.to(self.device).eval()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            self.get_logger().warning(
                f'Failed to move model to {self.device}: {error}. Falling back to CPU.'
            )
            if self.device.type == 'cuda':
                self._safe_empty_cuda_cache()
            self.device = torch.device('cpu')
            gc.collect()
            # Rebuild on CPU because a failed CUDA transfer can leave the module in an invalid state.
            model = self._build_model_from_checkpoint(checkpoint, GSCNN)
            model = model.to(self.device).eval()
            self.get_logger().info('Model loaded on CPU instead.')
        else:
            if self.device.type == 'cuda':
                if self.use_channels_last:
                    try:
                        model = model.to(memory_format=torch.channels_last)
                        self.get_logger().info('Using channels_last layout for the model')
                    except Exception as error:
                        self.get_logger().warning(
                            f'Failed to switch model to channels_last layout: {error}. Continuing with contiguous layout.'
                        )
                if self._use_mixed_precision():
                    self.get_logger().info('CUDA mixed precision enabled for inference')
                self._log_cuda_memory('after model load')

        return model

    def _build_model_from_checkpoint(self, checkpoint, gscnn_cls: type[torch.nn.Module]) -> torch.nn.Module:
        if isinstance(checkpoint, torch.nn.Module):
            return checkpoint.cpu().float().eval()

        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        model = gscnn_cls(num_classes=19, trunk='resnet50', criterion=None).eval()
        model.load_state_dict(state_dict, strict=False)
        return model.float().eval()

    def _use_mixed_precision(self) -> bool:
        return self.use_fp16 and self.device.type == 'cuda'

    def _safe_empty_cuda_cache(self):
        if self.device.type != 'cuda':
            return
        try:
            torch.cuda.empty_cache()
        except Exception as error:
            self.get_logger().warning(f'Failed to clear CUDA cache: {error}')

    def _log_cuda_memory(self, stage: str):
        if self.device.type != 'cuda':
            return
        try:
            total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            reserved_memory = torch.cuda.memory_reserved(self.device) / 1024**3
            allocated_memory = torch.cuda.memory_allocated(self.device) / 1024**3
            self.get_logger().info(
                f'GPU Memory ({stage}) - Total: {total_memory:.2f}GB, '
                f'Reserved: {reserved_memory:.2f}GB, Allocated: {allocated_memory:.2f}GB'
            )
        except Exception as error:
            self.get_logger().warning(f'Failed to query CUDA memory stats ({stage}): {error}')

    def _prepare_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).to(dtype=self.model_dtype) / 255.0
        tensor = tensor.unsqueeze(0).to(device=self.device, dtype=self.model_dtype)
        tensor = (tensor - self.mean) / self.std
        if self.use_channels_last:
            tensor = tensor.to(memory_format=torch.channels_last)
        return tensor

    def image_callback(self, msg: Image):
        now = time.perf_counter()
        if self.min_inference_period > 0.0 and (now - self.last_inference_time) < self.min_inference_period:
            return
        self.last_inference_time = now

        # Periodic garbage collection to prevent memory fragmentation
        self.frame_count += 1
        if self.frame_count % self.cleanup_interval == 0:
            gc.collect()
            if self.device.type == 'cuda':
                self._safe_empty_cuda_cache()

        inference_ms = None
        try:
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h0, w0 = image_bgr.shape[:2]

            # resize input if model was loaded with different dimensions to save GPU memory and speed up inference
            h1, w1 = self.resize_height, self.resize_width
            if (h1, w1) != (h0, w0):
                infer_bgr = cv2.resize(image_bgr, (w1, h1), interpolation=cv2.INTER_AREA)
            else:
                infer_bgr = image_bgr

            input_tensor = self._prepare_tensor(infer_bgr)
            autocast_context = (
                torch.autocast(device_type='cuda', dtype=torch.float16)
                if self._use_mixed_precision()
                else nullcontext()
            )

            start = time.perf_counter()
            with torch.inference_mode():
                with autocast_context:
                    output = self.model(input_tensor)

            inference_ms = (time.perf_counter() - start) * 1000

            logits = output[0] if isinstance(output, (tuple, list)) else output
            probabilities = None
            probability_classes = None
            if self.probabilities_pub is not None:
                probabilities = torch.softmax(logits, dim=1)
                probabilities, probability_classes = self._select_top_probabilities(probabilities)
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        except Exception as error:
            if self.device.type == 'cuda':
                self._safe_empty_cuda_cache()
            self.get_logger().error(f'Inference stage failed before publish: {error}')
            return

        try:
            if self.segmentation_pub is not None:
                segmentation_msg = self.bridge.cv2_to_imgmsg(prediction, encoding='mono8')
                segmentation_msg.header = msg.header
                self.segmentation_pub.publish(segmentation_msg)

            if probabilities is not None:
                probabilities_msg = self.bridge.cv2_to_imgmsg(
                    probabilities,
                    encoding=f'32FC{probabilities.shape[2]}',
                )
                probabilities_msg.header = msg.header
                self.probabilities_pub.publish(probabilities_msg)
                probability_classes_msg = self.bridge.cv2_to_imgmsg(
                    probability_classes,
                    encoding=f'8UC{probability_classes.shape[2]}',
                )
                probability_classes_msg.header = msg.header
                self.probability_classes_pub.publish(probability_classes_msg)

            if self.overlay_pub is not None:
                segmentation_rgb = self.class_colors_rgb[prediction]
                segmentation_bgr = cv2.cvtColor(segmentation_rgb, cv2.COLOR_RGB2BGR)
                if segmentation_bgr.shape[:2] != image_bgr.shape[:2]:
                    segmentation_bgr = cv2.resize(segmentation_bgr, (w0, h0), interpolation=cv2.INTER_NEAREST)
                overlay_bgr = cv2.addWeighted(image_bgr, 0.5, segmentation_bgr, 0.5, 0.0)
                overlay_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding='bgr8')
                overlay_msg.header = msg.header
                self.overlay_pub.publish(overlay_msg)
        except Exception as error:
            inference_text = 'unknown' if inference_ms is None else f'{inference_ms:.3f} ms'
            self.get_logger().error(
                f'Publish stage failed after successful inference ({inference_text}): {error}'
            )

    def _select_top_probabilities(self, probabilities: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Keep only thresholded top-k probabilities and their class ids."""
        probabilities = probabilities[0]
        num_classes = probabilities.shape[0]
        keep_k = min(self.max_probability_classes, num_classes)

        top_values, top_indices = torch.topk(probabilities, k=keep_k, dim=0)
        keep_mask = top_values > self.probability_threshold

        top_values = torch.where(keep_mask, top_values, 0.0)
        invalid_class_id = torch.full_like(top_indices, fill_value=255)
        top_indices = torch.where(keep_mask, top_indices, invalid_class_id)
        return (
            top_values.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32),
            top_indices.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8),
        )


def main():
    rclpy.init()
    node = SemanticSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
