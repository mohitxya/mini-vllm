import torch


def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version used by PyTorch:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:", torch.cuda.get_device_name(i))

        x = torch.randn(2048, 2048, device="cuda")
        y = torch.randn(2048, 2048, device="cuda")

        torch.cuda.synchronize()
        z = x @ y
        torch.cuda.synchronize()

        print("CUDA tensor test passed.")
        print("Result shape:", tuple(z.shape))

    else:
        print("No CUDA GPU visible to PyTorch.")


if __name__ == "__main__":
    main()