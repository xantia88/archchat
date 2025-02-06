import yaml

filename = "documents/standards/output.txt"

with open(filename, "r") as file, open("documents/standards.txt", "w") as out:
    objects = yaml.safe_load(file)
    arr = objects["seaf.change.requirements.dzo"]
    for k in arr:
        object = arr[k]
        text = object["statement"].strip()
        sber = object["sber"]
        levels = sber["applicability_level"]
        if not text.endswith("."):
            text = f"{text}."
        out.write(text)
        out.write(
            " Данное требования применяется для систем с уровнем критичности ")
        out.write(",".join(levels))
        out.write(".")
        out.write("\n")
