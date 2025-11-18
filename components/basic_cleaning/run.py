#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    with wandb.init(job_type="basic_cleaning") as run:
        run.config.update(args)

        # Download input artifact. This will also log that this script is using this
        # particular version of the artifact
        artifact_local_path = run.use_artifact(args.input_artifact).file()
        df = pandas.read_csv(artifact_local_path)
        df_cleaned = df.drop(df[(df.price < args.min_price) | (df.price > args.max_price)].index)
        # print(df['price'].describe())
        # print(df_cleaned['price'].describe())
        df_cleaned.to_csv(args.output_artifact, index=False)
        # Create a new artifact
        artifact = wandb.Artifact(name=args.output_artifact,
                                  type=args.output_type,
                                  description=args.output_description)
        artifact.add_file(args.output_artifact)
        run.log_artifact(artifact)
        artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="very basic data cleaning")
    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="name & version of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="name for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="used for artifact clustering in w&b",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="prices under this value are considered outliers and will be removed",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="prices above this value are considered outliers and will be removed",
        required=True
    )


    args = parser.parse_args()

    go(args)
