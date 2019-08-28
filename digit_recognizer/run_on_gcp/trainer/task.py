
import argparse
import json
import os
from . import model
import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        "--batch_size",
        help = "Batch size for training steps",
        type = int,
        default = 100
    )
    parser.add_argument(
        "--learning_rate",
        help = "Initial learning rate for training",
        type = float,
        default = 0.001
    )
    parser.add_argument(
        "--train_steps",
        help = "Steps to run the training job for. A step is one batch-size",
        type = int,
        default = 0
    )
    parser.add_argument(
        "--output_dir",
        help = "GCS location to write checkpoints and export models",
        required = True
    )
    parser.add_argument(
        "--input_dir",
        help = "GCS location to read input data",
        required = True
    )
    # Generate list of model functions to print in help message
    model_names = [name.replace("_model","") for name in dir(model) if name.endswith("_model")]
    parser.add_argument(
        "--model",
        help = "Type of model. Supported types are {}".format(model_names),
        default = "linear"
    )
    parser.add_argument(
        "--job-dir",
        help = "this model ignores this field, but it is required by gcloud",
        default = "junk"
    )
    parser.add_argument(
        "--dprob", 
        help = "dropout probability for CNN", 
        type = float, 
        default = 0.25
    )
    parser.add_argument(
        "--batch_norm", 
        help = "if specified, do batch_norm for CNN", 
        dest = "batch_norm", 
        action = "store_true"
    )
    ###### epochs #####
    parser.add_argument(
        "--epochs", 
        help = "number of epochs to train", 
        type = int, 
        default = 20
    )
   
    ###### decay step #####
    parser.add_argument(
        "--decay_steps", 
        help = "lr decay every epoch, decay_step = sample # // batch size", 
        type = float, 
        default = 300
    )
    #####################
    parser.set_defaults(batch_norm = False)

    args = parser.parse_args()
    hparams = args.__dict__

    # unused args provided by service
    hparams.pop("job_dir", None)
    hparams.pop("job-dir", None)

    output_dir = hparams.pop("output_dir")
    input_dir = hparams.pop("input_dir")
    # Append trial_id to path so hptuning jobs don"t overwrite eachother
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get("TF_CONFIG", "{}")
        ).get("task", {}).get("trial", "")
    )

    # Calculate train_steps if not provided
    if hparams["train_steps"] < 1:
        # 10,000 steps at batch_size of 512
        hparams["train_steps"] = (10000 * 100) // hparams["batch_size"]
        #print("Training for {} steps".format(hparams["train_steps"]))


    # Run the training job
    model.train_and_evaluate(output_dir, input_dir, hparams)