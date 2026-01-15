import json
import sys

def remove_int_id(data):
    keys_to_remove = {"int_id", "code", "icon_color", "background_color", "id", "type"}
    return [
        {k: v for k, v in obj.items() if k not in keys_to_remove}
        for obj in data
    ]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_int_id.py input.json output.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = remove_int_id(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"Done ✅ Saved cleaned JSON to {output_file}")
