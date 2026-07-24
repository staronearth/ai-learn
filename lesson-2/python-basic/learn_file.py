
from turtle import width


if __name__ == "__main__":
    with open("file.txt", "w") as f:
        f.write("Hello, World!\n")
        f.write("Hello, World1!\n")
        f.write("Hello, World2!\n")
    with open("file.txt", "r") as f:
        content = f.read()
        print(content)

    with open("file.txt", "r") as f:
        for line in f:
            print(line.strip())
    chunk_size = 1024
    with open("file.txt", "r") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            print(chunk)
