"""pouringwater_generated dataset builder."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


class PouringwaterGenerated(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for pouringwater_generated dataset.
    
    数据格式与 LIBERO 完全一致：
      - observation/image:       (H, W, 3) uint8  — agentview 相机
      - observation/wrist_image: (H, W, 3) uint8  — 腕部相机
      - observation/state:       (8,) float32      — eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
      - action:                  (7,) float32      — delta_pos(3) + delta_axisangle(3) + gripper(1)
      - language_instruction:    string            — 任务指令
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release — generated from PouringWater environment.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (used by RLDS)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format="png",
                            doc="Main camera RGB observation (agentview).",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format="png",
                            doc="Wrist camera RGB observation.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc="Robot state: eef_pos(3) + eef_axisangle(3) + gripper_qpos(2).",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc="Robot action: delta_pos(3) + delta_axisangle(3) + gripper(1).",
                    ),
                    "reward": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Reward — 1.0 at terminal step, 0.0 otherwise.",
                    ),
                    "discount": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Discount factor, always 1.0.",
                    ),
                    "is_first": tf.bool,
                    "is_last": tf.bool,
                    "is_terminal": tf.bool,
                    "language_instruction": tfds.features.Text(
                        doc="Language instruction for the task.",
                    ),
                }),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(path=self.data_dir),
        }

    def _generate_examples(self, path):
        """
        Generator of examples for each split.
        
        Note: 当数据已经通过 RLDSDataWriter 写入 TFRecord 时，
        这个方法不会被调用（TFDS 直接从 TFRecord 读取）。
        这里提供一个空实现，仅在需要从原始数据重新生成时使用。
        """
        return
        yield  # make it a generator
