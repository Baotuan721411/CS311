import argparse
import json
import os

from src.Retriever import BM25Retriever, SBertRetriever, HybridRetriever
from src.Trainer import LoRATrainer


def build_retriever(name: str, config: dict):
    if name == "SBertRetriever":
        return SBertRetriever(**config)
    elif name == "BM25Retriever":
        return BM25Retriever(**config)
    elif name == "HybridRetriever":
        return HybridRetriever(**config)
    else:
        raise ValueError(f"Unknown retriever: {name}")

def main():
    parser = argparse.ArgumentParser(
        description="Unified Runner (Retriever | LoRA Training | Evaluation)"
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["retrieve", "train", "eval"],
        required=True,
        help="Pipeline stage to run"
    )

    parser.add_argument(
        "--retriever",
        type=str,
        choices=["SBertRetriever", "BM25Retriever", "HybridRetriever"],
        default="SBertRetriever",
        help="Retriever type (only used when stage=retrieve)"
    )

    parser.add_argument(
        "--lora_config",
        type=str,
        help="Path to LoRA/LLM config (used for train/eval)"
    )

    args = parser.parse_args()

    if args.stage == "retrieve":
        with open("config/RetrieverConfig.json", "r", encoding="utf-8") as f:
            retriever_config = json.load(f)

        retriever = build_retriever(args.retriever, retriever_config)
        retriever.retrieve()

    elif args.stage == "train":
        if args.lora_config is None:
            raise ValueError("--lora_config is required for training")

        trainer = LoRATrainer(args.lora_config)
        trainer.train()

    elif args.stage == "eval":
        if args.lora_config is None:
            raise ValueError("--lora_config is required for evaluation")

        with open(args.lora_config, "r", encoding="utf-8") as f:
            lora_cfg = json.load(f)

        output_dir = lora_cfg["SFT_CONFIG"]["output_dir"]

        if (not os.path.exists(output_dir)) or (len(os.listdir(output_dir)) == 0):
            raise RuntimeError(
                f"Output directory '{output_dir}' is empty. "
                "Please run training before evaluation."
            )

        trainer = LoRATrainer(args.lora_config)
        trainer.evaluate()

if __name__ == "__main__":
    main()
