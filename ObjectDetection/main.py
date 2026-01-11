# from lightning.pytorch.cli import LightningCLI
from cli import ODLightningCLI

def cli_main():
    # Use ODLightningCLI instead of LightningCLI to main compatibility for .yaml 
    cli = ODLightningCLI()

if __name__ == "__main__":
    cli_main()

