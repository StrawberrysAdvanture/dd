import torch, os, json
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task, wan_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import numpy as np
import torchvision.transforms.functional as TF
import re
from transformers import pipeline
ocr = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=0)



def compute_ocr_reward(video_frames, target_text):
    detected = ""
    for frame in video_frames:
        result = ocr(frame.convert("RGB"))[0]["generated_text"].upper()
        detected += result
    # Count matching characters
    count = sum(1 for c in target_text if c in detected)
    return count / max(len(target_text), 1)



def extract_quoted_text(prompt):
    match = re.search(r"'(.*?)'", prompt)
    return match.group(1).strip().upper() if match else ""
 

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        
        ########################################## modifications happened here
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        video = data["video"]
        video_tensor = torch.stack([TF.to_tensor(frame) for frame in video])  # (C, T, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        video_tensor = video_tensor.to(dtype=torch.bfloat16, device=self.pipe.device)
        latent = self.pipe.vae.encode(video_tensor, device = self.pipe.device)
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "input_latents" : latent  ##########################################################################added input latents here
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    '''
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss
    '''
   ##################################################################################################################################################
    def forward(self, data, inputs=None):
        prompt = data["prompt"]
        target_text = extract_quoted_text(prompt)
        group_size = 4               # You can make this configurable
        temperature = 0.5            # Same here
        alpha = 0.8                  # OCR reward weight
        beta = 0.2                   # Optional: log_prob weight

        seeds = [random.randint(0, 999999) for _ in range(group_size)]

        rewards = []
        log_probs = []
        losses = []

        for seed in seeds:
            # 1. Sample video and get log_prob (-loss)
            video, log_prob = self.pipe.sample_video_with_logprob(
                prompt=prompt,
                height=480,
                width=832,
                num_frames=81,
                seed=seed,
            )

            # 2. Compute OCR reward
            ocr_reward = compute_ocr_reward(video, target_text)
            total_reward = alpha * ocr_reward + beta * log_prob.item()
            # 3. Re-evaluate training loss on this video
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            inputs_i = self.forward_preprocess({"prompt": prompt, "video": video})
            loss_i = self.pipe.training_loss(**models, **inputs_i)
            log_probs.append(log_prob)

            rewards.append(total_reward)
            losses.append(loss_i)

        # 4. Compute GRPO-weighted loss
        #rewards_tensor = torch.tensor(rewards, device=losses[0].device)
        #weights = torch.softmax(rewards_tensor / temperature, dim=0)
        #loss_tensor = torch.stack(losses)
        #final_loss = torch.sum(weights * loss_tensor)
        
        rewards_tensor = torch.tensor(rewards, device=losses[0].device)
    	advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

    	# === Step 2: GRPO-style clipped surrogate objective ===
    	log_probs_tensor = torch.stack(log_probs)
    	log_probs_old = log_probs_tensor.detach()  # Freeze old policy

    	ratios = torch.exp(log_probs_tensor - log_probs_old)  # Should be ~1.0 if stable
    	clipped_ratios = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps)
    
    	policy_loss = -torch.mean(torch.min(ratios * advantages, clipped_ratios * advantages))

        return policy_loss




if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = VideoDataset(args=args)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
