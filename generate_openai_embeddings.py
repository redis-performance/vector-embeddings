import pandas as pd
import multiprocessing as mp
import tqdm
from openai import OpenAI
import numpy as np
import os
import argparse
import multiprocessing
import time
import random


def get_embedding_with_retry(
    client,
    cleaned_input,
    embedding_dimension,
    model="text-embedding-3-large",
    max_retries=3,
):
    """Get embeddings with retry logic for handling API errors."""
    for attempt in range(max_retries):
        try:
            embeddings = []
            openai_reply = client.embeddings.create(
                input=cleaned_input,
                model=model,
                encoding_format="float",
                dimensions=embedding_dimension,
            )
            for row in openai_reply.data:
                assert len(row.embedding) == embedding_dimension
                embeddings.append(row.embedding)

            assert len(cleaned_input) == len(embeddings)
            return embeddings

        except Exception as e:
            wait_time = (2**attempt) + random.uniform(
                0, 1
            )  # Exponential backoff with jitter
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed")
                raise e


def get_embedding(
    client, cleaned_input, embedding_dimension, model="text-embedding-3-large"
):
    """Legacy function for backward compatibility."""
    return get_embedding_with_retry(client, cleaned_input, embedding_dimension, model)


def get_cleaned_input(df):
    titles = df["title"]
    texts = df["text"]
    cleaned_input = []
    cleaned_titles = []
    cleaned_texts = []
    for title in titles:
        cleaned_titles.append(f"{title}".replace("\n", " "))
    for text in texts:
        cleaned_texts.append(f"{text}".replace("\n", " "))
    for pos, cleaned_title in enumerate(cleaned_titles):
        cleaned_text = cleaned_texts[pos]
        cleaned_input.append(f"{cleaned_title} {cleaned_text}")

    assert len(cleaned_input) == len(titles)
    assert len(cleaned_input) == len(texts)
    return cleaned_input


def process_frame(cleaned_input, row_start, embedding_model, embedding_dimension):
    total_rows = len(cleaned_input)
    embeddings_filename = f"output/embedded_dbpedia_1M_dim_{embedding_dimension}_{row_start}_{total_rows}.npy"

    # Check if already completed
    if os.path.exists(embeddings_filename):
        print(f"✅ Chunk {row_start}: already exists, skipping...")
        return {
            "success": True,
            "rows": total_rows,
            "chunk_start": row_start,
            "filename": embeddings_filename,
        }

    try:
        print(f"🔄 Processing chunk {row_start} ({total_rows} rows)...")

        # Create OpenAI client inside the process to avoid serialization issues
        client = OpenAI()

        # Process data frame with retry logic
        embeddings = get_embedding_with_retry(
            client,
            cleaned_input,
            embedding_dimension,
            model=embedding_model,
            max_retries=3,
        )

        # Save embeddings atomically (write to temp file first, then rename)
        temp_filename = f"{embeddings_filename}.tmp"
        with open(temp_filename, "wb") as f:
            np.save(f, np.array(embeddings))

        # Atomic rename (prevents partial files)
        os.rename(temp_filename, embeddings_filename)

        assert len(cleaned_input) == len(embeddings)
        print(
            f"✅ Chunk {row_start}: successfully processed and saved to {embeddings_filename}"
        )

        return {
            "success": True,
            "rows": total_rows,
            "chunk_start": row_start,
            "filename": embeddings_filename,
        }

    except Exception as e:
        error_msg = f"❌ Chunk {row_start}: failed with error: {str(e)}"
        print(error_msg)

        # Clean up any partial files
        for temp_file in [embeddings_filename, f"{embeddings_filename}.tmp"]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"🧹 Cleaned up partial file: {temp_file}")
                except:
                    pass

        # Return error info instead of raising (keeps other processes running)
        return {"success": False, "rows": 0, "chunk_start": row_start, "error": str(e)}


def save_progress_report(successful_chunks, failed_chunks, embedding_dimension):
    """Save a progress report to help with retries and debugging."""
    import json
    from datetime import datetime

    report = {
        "timestamp": datetime.now().isoformat(),
        "embedding_dimension": embedding_dimension,
        "summary": {
            "successful_chunks": len(successful_chunks),
            "failed_chunks": len(failed_chunks),
            "total_rows_processed": sum(chunk["rows"] for chunk in successful_chunks),
        },
        "successful_chunks": successful_chunks,
        "failed_chunks": failed_chunks,
    }

    report_filename = f"output/progress_report_dim_{embedding_dimension}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        with open(report_filename, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📄 Progress report saved to: {report_filename}")
    except Exception as e:
        print(f"⚠️  Could not save progress report: {e}")


def get_failed_chunks_from_report(embedding_dimension):
    """Load failed chunks from the most recent progress report."""
    import json
    import glob

    try:
        # Find the most recent progress report
        pattern = f"output/progress_report_dim_{embedding_dimension}_*.json"
        reports = glob.glob(pattern)
        if not reports:
            return []

        latest_report = max(reports)
        with open(latest_report, "r") as f:
            report = json.load(f)

        failed_chunks = report.get("failed_chunks", [])
        if failed_chunks:
            print(
                f"📄 Found {len(failed_chunks)} failed chunks from previous run: {latest_report}"
            )

        return failed_chunks
    except Exception as e:
        print(f"⚠️  Could not load progress report: {e}")
        return []


if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)

    parser = argparse.ArgumentParser(
        prog="text-embedding-3-large embeddings generator",
    )
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--nrows", type=int, default=10000)
    parser.add_argument("--chunksize", type=int, default=50)
    parser.add_argument("--nprocesses", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-large")
    parser.add_argument("--embedding_dimension", type=int, default=3072)
    parser.add_argument("--csv_filename", type=str, default="input/dbpedia_1M.csv")
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Retry only failed chunks from previous run",
    )

    args = parser.parse_args()
    skiprows = args.skiprows
    nrows = args.nrows
    chunksize = args.chunksize
    nprocesses = args.nprocesses
    embedding_model = args.embedding_model
    embedding_dimension = args.embedding_dimension
    csv_filename = args.csv_filename
    retry_failed = args.retry_failed

    if not os.path.exists("output"):
        os.makedirs("output")

    # Handle retry mode
    if retry_failed:
        failed_chunks = get_failed_chunks_from_report(embedding_dimension)
        if not failed_chunks:
            print("🎉 No failed chunks found to retry!")
            exit(0)

        print(f"🔄 Retry mode: Processing {len(failed_chunks)} failed chunks...")
        # TODO: Implement retry logic for specific chunks
        # For now, continue with normal processing which will skip completed chunks
    else:
        print(
            f"🚀 Normal mode: Processing {nrows} rows starting from position {skiprows}..."
        )

    print(f"Splitting work among {nprocesses}...")
    print(
        f"Will read {nrows} rows from {csv_filename} starting a position {skiprows}..."
    )
    reader = pd.read_csv(csv_filename, chunksize=chunksize, nrows=nrows + skiprows)
    pool = mp.Pool(nprocesses)

    funclist = []
    at_row = 0
    print(f"creating chunks of work. each chunk has {chunksize} rows...")
    bar = tqdm.tqdm(total=nrows)
    for df in reader:
        if at_row < skiprows:
            at_row = at_row + len(df)
            continue
        else:
            cleaned_input = get_cleaned_input(df)
            # process each data frame
            f = pool.apply_async(
                process_frame,
                [cleaned_input, at_row, embedding_model, embedding_dimension],
            )
            funclist.append(f)
            bar.update(len(cleaned_input))
            at_row = at_row + len(df)
    bar.close()
    print(f"\n🚀 Processing {len(funclist)} chunks with {nprocesses} processes...\n")

    successful_chunks = []
    failed_chunks = []
    total_processed_rows = 0

    # Progress tracking
    bar = tqdm.tqdm(total=len(funclist), desc="Chunks processed")

    # Process all chunks and collect results
    for i, f in enumerate(funclist):
        try:
            # Get result with timeout
            result = f.get(timeout=600)  # 10 minute timeout per chunk

            if result["success"]:
                successful_chunks.append(result)
                total_processed_rows += result["rows"]
                bar.set_postfix(
                    {"✅": len(successful_chunks), "❌": len(failed_chunks)}
                )
            else:
                failed_chunks.append(result)
                bar.set_postfix(
                    {"✅": len(successful_chunks), "❌": len(failed_chunks)}
                )

        except Exception as e:
            # Handle timeout or other process errors
            error_result = {
                "success": False,
                "rows": 0,
                "chunk_start": f"unknown_{i}",
                "error": f"Process error: {str(e)}",
            }
            failed_chunks.append(error_result)
            print(f"❌ Process {i} failed: {e}")
            bar.set_postfix({"✅": len(successful_chunks), "❌": len(failed_chunks)})

        bar.update(1)

    pool.close()
    pool.join()
    bar.close()

    # Save progress report
    save_progress_report(successful_chunks, failed_chunks, embedding_dimension)

    # Print summary
    print(f"\n📊 Processing Summary:")
    print(f"✅ Successful chunks: {len(successful_chunks)}")
    print(f"❌ Failed chunks: {len(failed_chunks)}")
    print(f"📈 Total rows processed: {total_processed_rows}")

    if failed_chunks:
        print(f"\n⚠️  Failed chunks can be retried by running the script again.")
        print(
            f"💡 Consider using smaller --chunksize or fewer --nprocesses for better reliability."
        )

        # Show failed chunk details
        print(f"\n❌ Failed chunk details:")
        for chunk in failed_chunks[:5]:  # Show first 5 failures
            print(f"   Chunk {chunk['chunk_start']}: {chunk['error']}")
        if len(failed_chunks) > 5:
            print(f"   ... and {len(failed_chunks) - 5} more failures")
    else:
        print(f"\n🎉 All chunks processed successfully!")

    print(
        f"\n📁 Output files saved in: output/embedded_dbpedia_1M_dim_{embedding_dimension}_*.npy"
    )
