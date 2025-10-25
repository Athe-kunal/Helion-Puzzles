import torch
import helion.language as hl
import helion


@helion.kernel(autotune_effort="none")
def test_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] * 2
    return out


if __name__ == "__main__":
    x = torch.randn(100, device="cuda")
    result = test_kernel(x)
    torch.testing.assert_close(result, x * 2)
    print("Verification successful!")
