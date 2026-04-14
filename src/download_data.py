from src.data import download_dataset


def main():
    path = download_dataset(force=False)
    print(f"Done. Dataset available at: {path}")


if __name__ == "__main__":
    main()