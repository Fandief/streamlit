from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
from pathlib import Path


DEFAULT_CSV_FILES = [
    "all_platforms_classified_final.csv",
    "playstore_classified_final.csv",
    "reddit_classified_final.csv",
    "youtube_classified_final.csv",
]
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_payload(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"


def compress_csv(source: Path, level: int, force: bool) -> dict[str, str | int | bool]:
    destination = source.with_suffix(source.suffix + ".gz")
    if destination.exists() and not force:
        raise FileExistsError(f"{destination.name} sudah ada. Pakai --force untuk membuat ulang.")

    source_hash = sha256_file(source)
    with source.open("rb") as src, destination.open("wb") as raw_out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, compresslevel=level, mtime=0) as gz_out:
            shutil.copyfileobj(src, gz_out, length=CHUNK_SIZE)

    decompressed_hash = sha256_gzip_payload(destination)
    verified = source_hash == decompressed_hash
    if not verified:
        destination.unlink(missing_ok=True)
        raise ValueError(f"Verifikasi gagal untuk {source.name}; file .gz dihapus.")

    source_size = source.stat().st_size
    compressed_size = destination.stat().st_size
    saved_pct = (1 - compressed_size / source_size) * 100 if source_size else 0
    return {
        "source": source.name,
        "destination": destination.name,
        "source_size": source_size,
        "compressed_size": compressed_size,
        "saved_pct": round(saved_pct, 2),
        "verified": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kompres CSV besar menjadi .csv.gz secara lossless dan verifikasi isi datanya."
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_CSV_FILES,
        help="File CSV yang akan dikompres. Default: semua CSV dataset proyek ini.",
    )
    parser.add_argument("--level", type=int, default=9, choices=range(1, 10), help="Level kompresi gzip 1-9.")
    parser.add_argument("--force", action="store_true", help="Tulis ulang file .gz jika sudah ada.")
    args = parser.parse_args()

    print("Lossless CSV compression (.csv -> .csv.gz)")
    print("File asli tidak dihapus oleh script ini.")
    print()

    results = []
    for file_name in args.files:
        source = Path(file_name)
        if not source.exists():
            print(f"SKIP  {source.name}: file tidak ditemukan")
            continue
        if source.suffix.lower() != ".csv":
            print(f"SKIP  {source.name}: bukan file .csv")
            continue

        print(f"COMPRESS  {source.name} ...")
        result = compress_csv(source, level=args.level, force=args.force)
        results.append(result)
        print(
            "OK        "
            f"{result['destination']} | "
            f"{format_size(int(result['source_size']))} -> "
            f"{format_size(int(result['compressed_size']))} | "
            f"hemat {result['saved_pct']}% | verified"
        )

    if results:
        total_source = sum(int(item["source_size"]) for item in results)
        total_compressed = sum(int(item["compressed_size"]) for item in results)
        total_saved = (1 - total_compressed / total_source) * 100 if total_source else 0
        print()
        print(
            "TOTAL     "
            f"{format_size(total_source)} -> {format_size(total_compressed)} | "
            f"hemat {total_saved:.2f}%"
        )


if __name__ == "__main__":
    main()
