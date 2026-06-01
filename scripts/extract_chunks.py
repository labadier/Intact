from unstructured.partition.pdf import partition_pdf
from glob import glob
from pathlib import Path
from tqdm import tqdm
import dotenv
import json

from fire import Fire

dotenv.load_dotenv()

def document_parsing(data_path: Path):
    chunk_list = []

    for document in tqdm(glob(str(data_path / "*.pdf"))):
        chunks = partition_pdf(
            filename=document,
            languages=["eng"],
            infer_table_structure=False,
            extract_images_in_pdf = False,
            # strategy="hi_res",
            # Image extraction disabled for now
            # extract_image_block_types=["Image"],
            # extract_image_block_to_payload=True,
            chunking_strategy="basic",
            max_characters=3000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=3000,
        )
        print(f"Processed {Path(document).name} with {len(chunks)} chunks.")

        chunk_list += [
            {"text": chunk.text, "document": str(document)}
            for chunk in chunks
            if "CompositeElement" in str(type((chunk)))
        ]

    return chunk_list


def main(data_path: str, output_path: str) -> None:
    chunks = document_parsing(Path(data_path))

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "chunks.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(chunks)} chunks to {output_file}")
    return output_file


if __name__ == "__main__":
    Fire(main)
