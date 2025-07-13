import argparse


def add_io_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """File-system input/output."""
    group = parser.add_argument_group("I/O")
    group.add_argument("--data_dir", type=str, default="./data/Auditory cortex data", help="Directory containing .mat files")
    group.add_argument("--save_dir", type=str, default="./results", help="Directory to save results and checkpoints")
    return parser


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Model architecture & regularisation."""
    group = parser.add_argument_group("Model")
    group.add_argument(
        "--model_type",
        type=str,
        default="QGCN",
        choices=["GCN", "GAT", "AttentionGNN", "QGCN", "HyperGNN"],
        help="Type of (quantum) GNN model to use",
    )
    group.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    group.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers")
    group.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    return parser


def add_optim_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Optimiser hyper‑parameters."""
    group = parser.add_argument_group("Optimisation")
    group.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    group.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    group.add_argument("--batch_size", type=int, default=32, help="Mini‑batch size")
    group.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
    return parser


def add_graph_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Graph‑construction options."""
    group = parser.add_argument_group("Graph construction")
    group.add_argument("--k", type=int, default=20, help="k‑NN connectivity for functional graphs")
    group.add_argument("--connectivity_threshold", type=float, default=0.5, help="Threshold for functional connectivity")
    group.add_argument("--add_hyperedges", type=bool, default=False, help="Whether to add hyperedges to the graph")
    return parser


def add_checkpoint_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Checkpointing & evaluation."""
    group = parser.add_argument_group("Checkpointing / evaluation")
    group.add_argument("--checkpoint_freq", type=int, default=5, help="Save checkpoint every N epochs")
    group.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from")
    group.add_argument("--evaluate_only", type=str, default=None, help="Path to checkpoint for evaluation only")
    return parser


def add_misc_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Miscellaneous settings."""
    group = parser.add_argument_group("Misc")
    group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    group.add_argument("--col_width", type=int, default=50, help="Column width for anatomical dataset")
    group.add_argument("--col_height", type=int, default=30, help="Column height for anatomical dataset")
    group.add_argument(
        "--dataset",
        type=str,
        default="functional",
        choices=["functional", "anatomical"],
        help="Type of dataset to use",
    )
    return parser


def add_quantum_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Quantum‑specific arguments (only used when `--model_type QGNN`)."""
    group = parser.add_argument_group("Quantum (only for QGNN)")
    group.add_argument(
        "--quantum_device",
        type=str,
        default='default',
        choices=["ionq_aria", "ionq_forte", "default", "ionq_sim"],
        help="Quantum backend name (e.g. 'ibmq_qasm_simulator', 'ionq_qpu')",
    )
    group.add_argument("--shots", type=int, default=None, help="Number of circuit executions per job (shots)")
    group.add_argument("--api_key", type=str, default=None, help="Provider API key/token")
    group.add_argument("--q_depths", type=list, nargs="+", default=[2, 2], help="Quantum circuit depths for each layer")
    return parser

def build_arg_parser() -> argparse.ArgumentParser:
    """Factory that assembles the full parser from atomic groups."""
    parser = argparse.ArgumentParser(
        description="Neural Graph Classification with (Quantum) GNNs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for add_group in (
        add_io_args,
        add_model_args,
        add_optim_args,
        add_graph_args,
        add_checkpoint_args,
        add_misc_args,
        add_quantum_args,
    ):
        add_group(parser)
    return parser


def _validate_quantum_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Enforce that the QPU‑related arguments are provided when needed."""
    if args.model_type == "QGCN": # Make sure this matches your model type name
        missing = []
        # Check if quantum_device is required and missing
        if args.quantum_device is None or args.quantum_device == 'default':
            if args.quantum_device not in ['default', 'ionq_sim'] and args.api_key is None and os.getenv("QBRAID_API_KEY") is None:
                missing.append("--api_key (or QBRAID_API_KEY env var)")
            if args.quantum_device is None:
                missing.append("--quantum_device")

        if args.shots is None:
            missing.append("--shots")

        if missing:
            parser.error(
                "When --model_type is 'QGCN', the following arguments are required: "
                + ", ".join(missing)
            )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def define_parameters() -> None:
    parser = build_arg_parser()
    return parser

if __name__ == "__main__":
    parser = define_parameters()
    args = parser.parse_args()
    _validate_quantum_args(args, parser)
    print("Arguments parsed successfully!")