import os
import subprocess
from pathlib import Path
from typing import Optional, List


class KaggleDatasetDownloader:
    """Download datasets from Kaggle using the kaggle CLI.

    Requires kaggle credentials at one of:
      - %USERPROFILE%\.kaggle\kaggle.json (Windows)
      - ~/.kaggle/kaggle.json
      - Path set by env var KAGGLE_CONFIG_DIR (containing kaggle.json)
    """

    def __init__(self, output_root: str = "ml_models/data/raw"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def ensure_local_kaggle_dir(local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        readme_path = local_dir / "README.txt"
        if not readme_path.exists():
            readme_path.write_text(
                "Place your kaggle.json API credentials in this folder.\n"
                "Get it from: https://www.kaggle.com/settings/account -> Create New API Token\n"
                "Filename must be kaggle.json with permissions 600 (on Unix).\n"
            )

    @staticmethod
    def has_kaggle_credentials() -> bool:
        # Check env var first
        cfg_dir = os.environ.get("KAGGLE_CONFIG_DIR")
        if cfg_dir:
            if Path(cfg_dir, "kaggle.json").exists():
                return True
        # Default locations
        home = Path.home()
        if (home / ".kaggle" / "kaggle.json").exists():
            return True
        if os.name == "nt":
            # Additional Windows check: %USERPROFILE%\.kaggle
            userprofile = os.environ.get("USERPROFILE", str(home))
            if Path(userprofile) and Path(userprofile, ".kaggle", "kaggle.json").exists():
                return True
        return False

    def _run(self, cmd: List[str], cwd: Optional[Path] = None) -> None:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=cwd)

    def download_dataset(self, kaggle_ref: str, subdir: str) -> Path:
        """Download a Kaggle dataset given its ref like 'owner/dataset'."""
        if not self.has_kaggle_credentials():
            raise RuntimeError(
                "Kaggle credentials not found. Add kaggle.json to ~/.kaggle or set KAGGLE_CONFIG_DIR."
            )

        target_dir = self.output_root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Use kaggle CLI
        self._run(["kaggle", "datasets", "download", "-d", kaggle_ref, "-p", str(target_dir), "-f", "", "--unzip".strip()])
        return target_dir

    def download_preconfigured(self) -> None:
        """Download common datasets configured in config/config.py (if present)."""
        # Defaults if config not available
        default_datasets = {
            "urls/phishing_urls": "akashkr/phishing-website-dataset",
            "emails/spam_emails": "uciml/sms-spam-collection-dataset",
            "transactions/fraud_transactions": "mlg-ulb/creditcardfraud",
        }

        # Attempt to import configured mappings
        try:
            from ml_models.config.config import DATA_SOURCES  # type: ignore

            kaggle_map = DATA_SOURCES.get("kaggle_datasets", {})
            # Map to our subdirs
            mapping = {
                "urls/phishing_urls": kaggle_map.get("phishing_urls", default_datasets["urls/phishing_urls"]),
                "emails/spam_emails": kaggle_map.get("spam_emails", default_datasets["emails/spam_emails"]),
                "transactions/fraud_transactions": kaggle_map.get("fraud_transactions", default_datasets["transactions/fraud_transactions"]),
            }
        except Exception:
            mapping = default_datasets

        for subdir, ref in mapping.items():
            try:
                print(f"\nDownloading Kaggle dataset '{ref}' to '{subdir}'...")
                self.download_dataset(ref, subdir)
                print(f"✓ Downloaded: {ref}")
            except Exception as e:
                print(f"✗ Failed to download {ref}: {e}")


if __name__ == "__main__":
    # Prepare a local .kaggle directory inside the repo for convenience
    local_cfg = Path(".kaggle")
    KaggleDatasetDownloader.ensure_local_kaggle_dir(local_cfg)
    print(f"Local Kaggle config directory prepared at: {local_cfg.resolve()}")
    print("If you don't have credentials, create them in Kaggle settings and place kaggle.json here.")

    # Try downloads
    downloader = KaggleDatasetDownloader()
    if downloader.has_kaggle_credentials():
        downloader.download_preconfigured()
    else:
        print("Kaggle credentials not found. Skipping downloads. See .kaggle/README.txt for instructions.")



