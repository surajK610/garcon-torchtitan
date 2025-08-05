from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_qwen3
from .infra.pipeline import pipeline_qwen3
from .model.args import TransformerModelArgs
from .model.model import Transformer
from .model.state_dict_adapter import Qwen3StateDictAdapter

__all__ = [
    "parallelize_qwen3",
    "pipeline_qwen3",
    "TransformerModelArgs", 
    "Transformer",
    "qwen3_configs",
]

qwen3_configs = {
    "1.7B": TransformerModelArgs(
        dim=2048,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=3.0,  # intermediate_size=6144, so 6144/2048=3.0
        multiple_of=256,
        vocab_size=151936,
        rope_theta=1000000,
        max_seq_len=40960,
    ),
}



register_train_spec(
    TrainSpec(
        name="qwen3",
        model_cls=Transformer,
        model_args=qwen3_configs,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_qwen3,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
)