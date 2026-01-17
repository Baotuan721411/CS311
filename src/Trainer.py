import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from src.eval import evaluate_all

import json
def train(config: dict):
    MODEL_NAME = config["MODEL_NAME"]
    LORA_CFG = config["LORA_CONFIG"]
    SFT_CFG = config["SFT_CONFIG"]

    print(f"--> Đang tải model: {MODEL_NAME}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.bfloat16
    )

    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=LORA_CFG["r"],
        lora_alpha=LORA_CFG["lora_alpha"],
        lora_dropout=LORA_CFG["lora_dropout"],
        bias=LORA_CFG["bias"],
        task_type=LORA_CFG["task_type"],
        target_modules=LORA_CFG["target_modules"],
    )

    dataset = load_dataset(
        "json",
        data_files=config["train_file"],
        split="train"
    )

    sft_config = SFTConfig(
        output_dir=SFT_CFG["output_dir"],
        num_train_epochs=SFT_CFG["num_train_epochs"],
        per_device_train_batch_size=SFT_CFG["per_device_train_batch_size"],
        gradient_accumulation_steps=SFT_CFG["gradient_accumulation_steps"],
        learning_rate=SFT_CFG["learning_rate"],
        bf16=SFT_CFG["bf16"],
        fp16=SFT_CFG["fp16"],
        logging_steps=SFT_CFG["logging_steps"],
        save_strategy=SFT_CFG["save_strategy"],
        optim=SFT_CFG["optim"],
        report_to=SFT_CFG["report_to"]
    )
    sft_config.max_seq_length = SFT_CFG["max_seq_length"]
    sft_config.packing = SFT_CFG["packing"]

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer,
    )

    print("--> Bắt đầu Training...")
    trainer.train()

    output_dir = SFT_CFG["output_dir"]
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"--> HOÀN TẤT! Adapter đã lưu tại: {output_dir}")
    
    
class LoRATrainer:
    def __init__(
        self,
        config_path: dict
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["MODEL_NAME"],
            quantization_config = self.bnb_config,
            device_map="auto",
            use_cache=False,
            torch_dtype=torch.bfloat16
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["MODEL_NAME"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.LORA_CFG = self.config["LORA_CONFIG"]
        self.SFT_CFG = self.config["SFT_CONFIG"]
        self.peft_config = LoraConfig(
            r = self.LORA_CFG["r"],
            lora_alpha = self.LORA_CFG["lora_alpha"],
            lora_dropout = self.LORA_CFG["lora_dropout"],
            bias = self.LORA_CFG["bias"],
            task_type = self.LORA_CFG["task_type"],
            target_modules = self.LORA_CFG["target_modules"],
        )
        
        self.sft_config = SFTConfig(
            output_dir = self.SFT_CFG["output_dir"],
            num_train_epochs = self.SFT_CFG["num_train_epochs"],
            per_device_train_batch_size = self.SFT_CFG["per_device_train_batch_size"],
            gradient_accumulation_steps = self.SFT_CFG["gradient_accumulation_steps"],
            learning_rate = self.SFT_CFG["learning_rate"],
            bf16 = self.SFT_CFG["bf16"],
            fp16 = self.SFT_CFG["fp16"],
            logging_steps = self.SFT_CFG["logging_steps"],
            save_strategy = self.SFT_CFG["save_strategy"],
            optim = self.SFT_CFG["optim"],
            report_to = self.SFT_CFG["report_to"]
        )
        self.sft_config.max_seq_length = self.SFT_CFG["max_seq_length"]
        self.sft_config.packing = self.SFT_CFG["packing"]
        dataset = load_dataset(
            "json",
            data_files=self.config["train_file"],
            split="train"
        )
        self.trainer = SFTTrainer(
            model = self.model,
            train_dataset = dataset,
            peft_config = self.peft_config,
            args = self.sft_config,
            processing_class=self.tokenizer,
        )
    def train (self):
        self.trainer.train()
        output_dir = self.SFT_CFG["output_dir"]
        self.trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)    
        print(f"--> HOÀN TẤT! Model đã lưu tại: {output_dir}")
    def evaluate (self):
        evaluate_all(self.config)