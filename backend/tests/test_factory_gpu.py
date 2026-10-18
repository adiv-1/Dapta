import unittest

from dapta.factory_gpu import restore_frozen_compute_dtype


class _DType:
    pass


class _Weight:
    def __init__(self, dtype, *, requires_grad=False):
        self.dtype = dtype
        self.requires_grad = requires_grad


class _Module:
    def __init__(self, name, dtype, *, requires_grad=False):
        self.name = name
        self.weight = _Weight(dtype, requires_grad=requires_grad)
        self.requested_dtype = None
        self.frozen = False

    def requires_grad_(self, value):
        self.frozen = not value
        self.weight.requires_grad = value
        return self

    def to(self, *, dtype):
        self.requested_dtype = dtype
        self.weight.dtype = dtype
        return self


class _Linear(_Module):
    pass


class _Torch:
    float32 = _DType()
    bfloat16 = _DType()

    class nn:
        Linear = _Linear


class FactoryGpuTests(unittest.TestCase):
    def test_restores_only_frozen_compute_layers_to_bfloat16(self):
        model = type("Model", (), {})()
        model.model = type("Core", (), {})()
        model.model.visual = _Module("visual", _Torch.float32)
        model.model.language_model = type("Language", (), {})()
        model.model.language_model.embed_tokens = _Module("embed", _Torch.float32)
        frozen_head = _Linear("lm_head", _Torch.float32)
        trainable_linear = _Linear("adapter", _Torch.float32, requires_grad=True)
        norm = _Module("norm", _Torch.float32)
        model.named_modules = lambda: [
            ("lm_head", frozen_head),
            ("adapter", trainable_linear),
            ("model.norm", norm),
        ]

        converted = restore_frozen_compute_dtype(model, _Torch)

        self.assertEqual(
            converted,
            ["lm_head", "model.language_model.embed_tokens", "model.visual"],
        )
        self.assertIs(frozen_head.weight.dtype, _Torch.bfloat16)
        self.assertIs(trainable_linear.weight.dtype, _Torch.float32)
        self.assertIs(norm.weight.dtype, _Torch.float32)
        self.assertTrue(model.model.visual.frozen)


if __name__ == "__main__":
    unittest.main()
