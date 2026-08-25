#!/usr/bin/env python3
import concurrent.futures
import os
import sys
import time
import urllib.request

def download_chunk(url, start_byte, end_byte, out_path, chunk_idx):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start_byte}-{end_byte}'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        with open(out_path, 'r+b') as f:
            f.seek(start_byte)
            copied = 0
            while True:
                buf = resp.read(1024 * 1024)
                if not buf:
                    break
                f.write(buf)
                copied += len(buf)
    return copied

def main():
    if len(sys.argv) < 3:
        print("Usage: fast_download.py <URL> <OUTPUT_PATH> [THREADS]")
        sys.exit(1)
    
    url = sys.argv[1]
    out_path = sys.argv[2]
    num_threads = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    # Get total file size
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_size = int(resp.headers.get('Content-Length', 0))

    if total_size == 0:
        # Fallback to GET
        req = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_range = resp.headers.get('Content-Range', '')
            if '/' in content_range:
                total_size = int(content_range.split('/')[-1])

    print(f"File size: {total_size / (1024**3):.2f} GB ({total_size} bytes)")
    
    # Pre-allocate output file
    temp_path = out_path + ".tmp"
    with open(temp_path, "wb") as f:
        f.truncate(total_size)

    chunk_size = total_size // num_threads
    chunks = []
    for i in range(num_threads):
        start = i * chunk_size
        end = (start + chunk_size - 1) if (i < num_threads - 1) else (total_size - 1)
        chunks.append((start, end, i))

    print(f"Downloading with {num_threads} parallel threads...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(download_chunk, url, start, end, temp_path, idx): idx
            for (start, end, idx) in chunks
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                copied = future.result()
            except Exception as exc:
                print(f"Chunk {idx} generated an exception: {exc}")
                sys.exit(1)

    elapsed = time.time() - start_time
    speed_mb = (total_size / (1024 * 1024)) / elapsed
    print(f"Downloaded {total_size / (1024**3):.2f} GB in {elapsed:.1f}s ({speed_mb:.1f} MB/s)")
    os.rename(temp_path, out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
