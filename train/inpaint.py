from trainer import main, parse_args

config = parse_args()
config.cache_dir = ".cache"
config.pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
# config.pretrained_controlnet_name_or_path = "lllyasviel/control_v11f1e_sd15_tile"
config.resolution = 512
config.control_lora_linear_rank = 32
config.control_lora_conv2d_rank = 32
config.learning_rate = 1e-4
config.train_batch_size = 4
config.max_train_steps = 75000
config.enable_xformers_memory_efficient_attention = False
config.checkpointing_steps = 250
config.validation_steps = 5000
config.report_to = "wandb"
config.resume_from_checkpoint = "latest"
# config.push_to_hub = True

config.custom_dataset = "custom_datasets.CO3D.WarpCO3DDataset"
config.conditioning_image_column = "condition"
config.image_column = "target"
config.caption_column = "prompt"
config.num_validation_samples = 3
config.conditioning_type_name = "correct-inpaint-v1"

config.tracker_project_name = f"sd-control-dora-{config.conditioning_type_name}"
config.output_dir = f"output/{config.tracker_project_name}"
config.use_dora = True

if __name__ == '__main__':
    main(config)
