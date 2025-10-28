"""Count parameters for large EALSTM and Mamba models (~1M params)."""

from transfer_learning_publication.models import ModelFactory


def count_parameters(model) -> dict[str, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: PyTorch Lightning model

    Returns:
        Dictionary with total and trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def test_ealstm_config(hidden_size: int, num_layers: int, bidirectional: bool = False) -> dict[str, int]:
    """Test EALSTM configuration and return parameter counts."""
    config = {
        "bidirectional": bidirectional,
        "dropout": 0.2,  # Placeholder
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "input_size": 6,  # 6 forcing features
        "static_size": 22,  # 22 static features
        "input_len": 100,
        "output_len": 1,
    }

    factory = ModelFactory()
    model = factory.create_from_dict("ealstm", config)
    return count_parameters(model)


def test_mamba_config(d_model: int, n_layers: int, d_state: int, decoder_hidden_size: int) -> dict[str, int]:
    """Test Mamba configuration and return parameter counts."""
    config = {
        "d_conv": 4,
        "d_model": d_model,
        "d_state": d_state,
        "decoder_hidden_size": decoder_hidden_size,
        "dropout": 0.2,  # Placeholder
        "expand": 2,
        "n_layers": n_layers,
        "input_size": 6,  # 6 forcing features
        "static_size": 22,  # 22 static features
        "input_len": 100,
        "output_len": 1,
    }

    factory = ModelFactory()
    model = factory.create_from_dict("mamba", config)
    return count_parameters(model)


def main() -> None:
    """Test different configurations to find ~1M parameter models."""
    target_params = 1_000_000

    print("=" * 80)
    print("SEARCHING FOR ~1M PARAMETER CONFIGURATIONS")
    print(f"Target: {target_params:,} parameters")
    print("=" * 80)

    # Test EALSTM configurations
    print("\n" + "=" * 80)
    print("EALSTM CONFIGURATIONS")
    print("=" * 80)

    ealstm_configs = [
        # (hidden_size, num_layers, bidirectional)
        # Refining around 300-350 with 3 layers
        (300, 3, False),
        (310, 3, False),
        (320, 3, False),
        (325, 3, False),
        (330, 3, False),
        (335, 3, False),
        (340, 3, False),
    ]

    best_ealstm = None
    best_ealstm_diff = float('inf')

    for hidden_size, num_layers, bidirectional in ealstm_configs:
        counts = test_ealstm_config(hidden_size, num_layers, bidirectional)
        diff = abs(counts['total'] - target_params)

        print(f"\nhidden_size={hidden_size}, num_layers={num_layers}, bidirectional={bidirectional}")
        print(f"  Total parameters: {counts['total']:,}")
        print(f"  Difference from target: {diff:,}")

        if diff < best_ealstm_diff:
            best_ealstm_diff = diff
            best_ealstm = (hidden_size, num_layers, bidirectional, counts['total'])

    # Test Mamba configurations
    print("\n" + "=" * 80)
    print("MAMBA CONFIGURATIONS")
    print("=" * 80)

    mamba_configs = [
        # (d_model, n_layers, d_state, decoder_hidden_size)
        # Need to increase parameters significantly
        (180, 4, 32, 250),
        (190, 4, 32, 250),
        (200, 4, 32, 250),
        (210, 4, 32, 250),
        (220, 4, 32, 250),
        (200, 4, 32, 200),
        (200, 4, 32, 220),
        (200, 4, 32, 230),
        (200, 4, 32, 240),
        (210, 4, 32, 200),
        (210, 4, 32, 220),
    ]

    best_mamba = None
    best_mamba_diff = float('inf')

    for d_model, n_layers, d_state, decoder_hidden_size in mamba_configs:
        counts = test_mamba_config(d_model, n_layers, d_state, decoder_hidden_size)
        diff = abs(counts['total'] - target_params)

        print(f"\nd_model={d_model}, n_layers={n_layers}, d_state={d_state}, decoder_hidden_size={decoder_hidden_size}")
        print(f"  Total parameters: {counts['total']:,}")
        print(f"  Difference from target: {diff:,}")

        if diff < best_mamba_diff:
            best_mamba_diff = diff
            best_mamba = (d_model, n_layers, d_state, decoder_hidden_size, counts['total'])

    # Summary
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)

    if best_ealstm:
        print(f"\nEALSTM:")
        print(f"  hidden_size: {best_ealstm[0]}")
        print(f"  num_layers: {best_ealstm[1]}")
        print(f"  bidirectional: {best_ealstm[2]}")
        print(f"  Total parameters: {best_ealstm[3]:,}")
        print(f"  Difference from target: {best_ealstm_diff:,}")

    if best_mamba:
        print(f"\nMamba:")
        print(f"  d_model: {best_mamba[0]}")
        print(f"  n_layers: {best_mamba[1]}")
        print(f"  d_state: {best_mamba[2]}")
        print(f"  decoder_hidden_size: {best_mamba[3]}")
        print(f"  Total parameters: {best_mamba[4]:,}")
        print(f"  Difference from target: {best_mamba_diff:,}")


if __name__ == "__main__":
    main()
