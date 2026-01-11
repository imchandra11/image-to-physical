# from lightning.pytorch.cli import LightningCLI
from cli import CRAFTLightningCLI

def cli_main():
    cli = CRAFTLightningCLI(run=False)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.get("fit.ckpt_path"))
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.get("test.ckpt_path"))
    # cli.trainer.predict(model=cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.get("test.ckpt_path"))

if __name__ == "__main__":
    cli_main()
