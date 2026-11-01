"""Quick functional example with no GPU or dataset download required."""
import torch
from fastesa import make_esa, make_model, forward_tokens

torch.manual_seed(123)
x = torch.randn(2, 65, 48)
for variant in ("fast", "quality"):
    mixer = make_esa(variant, dim=48, heads=6).eval()
    with torch.no_grad():
        full_y, full_state = mixer(x)
        first_y, first_state = mixer(x[:, :32])
        second_y, streamed_state = mixer(x[:, 32:], first_state)
        assert torch.allclose(torch.cat([first_y, second_y], 1), full_y, atol=5e-4, rtol=5e-4)
    model = make_model(variant, vocab_size=256, dim=48, heads=6, layers=1)
    logits, states = forward_tokens(model, torch.randint(0, 256, (2, 64)))
    print(variant, "mixer output", full_y.shape, "cache", full_state.shape,
          "last-token logits", logits.shape)
