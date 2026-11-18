from src.document_loader import file_loder

file_path = "data/raw/CNN_test_notes.pdf"

document = file_loder(file_path)

print(f"Total pages extracted: {len(document)}")

for text in document[:10]:
    print(f"--------{text['Page_Number']}----------------")
    print(text["Text"][:500])
