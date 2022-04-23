import os
import re
import readline
import subprocess
import sys
from pathlib import Path


class BadResponse(Exception):
    """User gave a bad response."""


def camel_to_snake(s):
    return "".join(["_" + c.lower() if c.isupper() else c for c in s]).lstrip("_")


def validate_input(prompt, validate=None):
    """Prompts user until response passes ``validate``."""
    while True:
        response = input(prompt + ": ")

        if validate is None:
            break

        try:
            validate(response)
            break
        except BadResponse as e:
            print(f'"{response}" {e}')
    return response


def main():
    def is_identifier(response):
        if not response.isidentifier():
            raise BadResponse("is not a valid python identifier.")

    def is_lower(response):
        if response.lower() != response:
            raise BadResponse("should be all lower case.")

    def good_module_name(response):
        is_identifier(response)
        is_lower(response)
        if "_" in response:
            raise BadResponse("should not contain an underscore _")
        if len(response) > 20:
            raise BadResponse("is too long (max 20 char limit).")

    developer_name = validate_input("Enter your name")

    module_name = validate_input("Python Module Name (default: app)", good_module_name)

    def good_class_name(response):
        is_identifier(response)
        if not response[0].isupper():
            raise BadResponse("first letter should be capitalized.")

    dataset_class_name = validate_input(
        "Dataset class name (default: MyDataset)", good_class_name
    )

    model_class_name = validate_input(
        "Model class name (default: MyModel)", good_class_name
    )

    replacements = {
        "YOUR_NAME_HERE": developer_name,
        "app": module_name,
        "MyDataset": dataset_class_name,
        "mydataset": dataset_class_name.lower(),
        "MyModel": model_class_name,
        "mymodel": model_class_name.lower(),
    }

    def replace(string):
        def _replace(match):
            return replacements[match.group(0)]

        # notice that the 'this' in 'thistle' is not matched
        return re.sub(
            "|".join(r"\b%s\b" % re.escape(s) for s in replacements),
            _replace,
            "the cat has this thistle.",
        )

    repo = Path(__file__).parent
    all_py_files = list(repo.rglob("*.py"))
    for py_file in all_py_files:
        contents = py_file.read_text()
        import ipdb

        ipdb.set_trace()

    # Delete this script at the end of execution
    # os.remove(sys.argv[0])


if __name__ == "__main__":
    main()
