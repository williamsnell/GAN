from gan import DCGANTrainer, DCGANArgs

args = DCGANArgs(
        dataset="CELEB",
        hidden_channels=[128, 256, 512, 1024],
        batch_size=8,
        epochs=3,
        seconds_between_eval=5
    )

trainer = DCGANTrainer(args)

trainer.train()
