from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_llama
from .model.args import TransformerModelArgs
from .model.model import Transformer
from .model.state_dict_adapter import Llama3StateDictAdapter

__all__ = [
    "parallelize_llama",
    "TransformerModelArgs",
    "Transformer",
    "llama3_args",
]


llama3_args = {
    "testmodel": TransformerModelArgs( # 460M
        dim=1024,
        n_layers=16,
        n_heads=8,
        n_kv_heads=4,
        ffn_dim_multiplier=1.1,
        multiple_of=512,
        rope_theta=500000,
    ),
    "llama-115M": TransformerModelArgs(
        dim=768,
        n_layers=10,
        n_heads=8,
        n_kv_heads=4,
        ffn_dim_multiplier=1.0,
        multiple_of=256,
        rope_theta=500000,
        vocab_size=32001,
        attn_type="sdpa",
        attn_mask_type="causal",
        max_seq_len=1536,
        eos_id=2
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args=llama3_args,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
