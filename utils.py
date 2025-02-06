import yaml

filename = "documents/standards/output.txt"

with open(filename, "r") as file, open("documents/standards.txt", "w") as out:
    objects = yaml.safe_load(file)
    arr = objects["seaf.change.requirements.dzo"]
    for k in arr:
        object = arr[k]
        text = object["statement"]
        out.write(text)
        out.write("\n")
