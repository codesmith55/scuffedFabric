import os; import psutil; import timeit

from datasets import load_dataset

# Process.memory_info is expressed in bytes, so convert to megabytes 

mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

wiki = load_dataset("wikipedia", "20220301.en", split="train")

mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

print(f"RAM memory used: {(mem_after - mem_before)} MB")