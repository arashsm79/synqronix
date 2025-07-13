"""
Call the training/evaluation pipeline with parameters that come from the
GitHub-Actions workflow dispatch.
"""
import os, sys, json
from src.synqronix.config import define_parameters
from src.synqronix.main import main, build_data

def cli():
    # GitHub passes all inputs as env-vars. Retrieve them or fall back.
    ginputs = json.loads(os.environ.get("QBRAID_RUN_INPUTS", "{}"))

    # Build the same parser you already use
    parser = define_parameters()
    cmdline_args = sys.argv[1:]    

    # Add / override with the UI inputs
    for k, v in ginputs.items():
        cmdline_args += [f"--{k}", str(v)]

    args = parser.parse_args(cmdline_args)
    (data_loader, train_loader, val_loader,
     test_loader, device, checkpoint_dir, eval_dir) = build_data(args)
    
    # Train classical GCN model
    args.model_type = "GCN"
    main(args, data_loader, train_loader, 
         val_loader, test_loader, device, 
         checkpoint_dir, eval_dir)  
    print("Training of GCN completed successfully!")

    # Train quantum GCN model
    args.model_type = "QGNN"
    args.shots = 1024
    args.api_key = os.environ.get("QBRAID_API_KEY", None)
    if not args.api_key:
        raise ValueError("QBRAID_API_KEY environment variable is not set.")
    
    main(args, data_loader, train_loader, 
         val_loader, test_loader, device, 
         checkpoint_dir, eval_dir)
    
    print("Training of QGNN completed successfully!")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli()
