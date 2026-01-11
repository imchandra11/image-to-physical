# from lightning.pytorch.cli import LightningCLI
from cli import EasyOCRLightningCLI

def cli_main():
    cli = EasyOCRLightningCLI()
    
if __name__ == "__main__":
    cli_main()











# import os
# from lightning.pytorch.cli import LightningCLI
# from module import EasyOCRLitModule
# from datamodule import EasyOCRDataModule

# def main():
#     # Automatically parses --config and runs fit/test/etc.
#     LightningCLI(
#         model_class=EasyOCRLitModule,
#         datamodule_class=EasyOCRDataModule,
#         save_config_callback=None,  # don't auto-overwrite config
#         run=True  # auto-call trainer.<fit/test/etc>
#     )

# if __name__ == "__main__":
#     main()
