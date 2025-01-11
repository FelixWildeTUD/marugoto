from fire import Fire

def extract():
    print("extract")

if __name__ == "__main__":
    print("main")
    Fire(
        {
            "extract": extract,
        }
    )
