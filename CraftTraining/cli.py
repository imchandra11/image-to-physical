from lightning.pytorch.cli import LightningCLI

class CRAFTLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # For RESUME
        parser.add_argument("--fit.ckpt_path")
        # Select last or best checkpoint
        parser.add_argument("--test.ckpt_path", default="best")
        
       
        
