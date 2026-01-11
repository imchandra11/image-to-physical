# from lightning.pytorch.cli import LightningCLI
from cli import ODLightningCLI

def cli_main():
    cli = ODLightningCLI(run=False)
    cli.trainer.fit(model = cli.model, datamodule = cli.datamodule, ckpt_path=cli.config['fit.ckpt_path'])
    cli.trainer.test(datamodule = cli.datamodule, ckpt_path=cli.config['test.ckpt_path'])

if __name__ == "__main__":
    cli_main()

