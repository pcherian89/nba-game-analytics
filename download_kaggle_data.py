import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_dataset(dataset_name: str, download_path: str = "nba_data"):
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")

    with open(kaggle_json_path, "w") as f:
        f.write(f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}')

    os.chmod(kaggle_json_path, 0o600)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset=dataset_name, path=download_path, unzip=True)

    print(f"✅ Dataset '{dataset_name}' downloaded and extracted to '{download_path}'")

if __name__ == "__main__":
    download_kaggle_dataset("eoinamoore/historical-nba-data-and-player-box-scores")
