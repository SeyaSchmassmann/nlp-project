import os
import datasets

currentFolder = os.path.dirname(os.path.realpath(__file__))

dataset = datasets.load_dataset("gxb912/large-twitter-tweets-sentiment")
dataset["train"].to_csv(os.path.join(currentFolder, "train.csv"), index=False)
dataset["test"].to_csv(os.path.join(currentFolder, "test.csv"), index=False)