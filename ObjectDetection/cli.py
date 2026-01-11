from lightning.pytorch.cli import LightningCLI

class ODLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # For RESUME
        parser.add_argument("--fit.ckpt_path")
        # Select last or best checkpoint
        parser.add_argument("--test.ckpt_path", default="best")
        # Set model classes equal to dataset calsses 
        parser.link_arguments(
            "data.init_args.classes", "model.init_args.num_classes", 
            compute_fn=lambda classes: len(classes)
        )
        
