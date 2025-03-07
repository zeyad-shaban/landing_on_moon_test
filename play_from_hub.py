from huggingface_sb3 import load_from_hub

checkpoint = load_from_hub(
	repo_id="bocchi-julia/ppo-LunarLander-v3",
	filename="ppo-LunarLander-v3.zip",
)