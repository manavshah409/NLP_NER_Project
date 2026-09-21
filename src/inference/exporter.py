"""In-memory UTF-8 downloads. User input is never written to disk."""
import csv
import io
import json


def json_export(result):
    return json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")


def csv_export(result):
    output = io.StringIO(newline="")
    fields = ["text", "type", "start_char", "end_char", "start_token", "end_token", "bio_sequence", "confidence"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for entity in result["entities"]:
        row = dict(entity, bio_sequence=" ".join(entity["bio_sequence"]))
        # Prevent spreadsheet formula execution; JSON retains the exact entity text.
        if row["text"].lstrip().startswith(("=", "+", "-", "@")):
            row["text"] = "'" + row["text"]
        writer.writerow(row)
    return output.getvalue().encode("utf-8-sig")
