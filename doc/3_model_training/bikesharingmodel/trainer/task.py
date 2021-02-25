import argparse
import json
import os

from . import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help = "Location directory to read datasets",
        type = str,
        required = True
    )
    parser.add_argument(
        "--output_dir",
        help = "Location directory to write checkpoints and export models",
        type = str,
        required = True
    )
    parser.add_argument(
        "--hidden_units_1",
        help = "Hidden layer 1 sizes to use for DNN feature columns",
        type = int,
        default = 16
    )
    parser.add_argument(
        "--hidden_units_2",
        help = "Hidden layer 2 sizes to use for DNN feature columns",
        type = int,
        default = 16
    )
    parser.add_argument(
        "--hidden_units_3",
        help = "Hidden layer 3 sizes to use for DNN feature columns",
        type = int,
        default = 16
    )
    parser.add_argument(
        "--batch_size",
        help = "Mini-batch gradient descent batch size",
        type = int,
        default = 16
    )
    parser.add_argument(
        "--num_epochs",
        help = "Number of epochs",
        type = int,
        default = 10
    )
    parser.add_argument(
        "--learning_rate",
        help = "Hidden layer 1 sizes to use for DNN feature columns",
        type = int,
        default = 0.001
    )
    args = parser.parse_args().__dict__


    # Append trial_id to path so trials don't overwrite each other
    args["output_dir"] = os.path.join(
        args["output_dir"],
        json.loads(
            os.environ.get("TF_CONFIG", "{}")
        ).get("task", {}).get("trial", "")
    ) 
    
    # Run the training job
    model.train_and_evaluate(args)
