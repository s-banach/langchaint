from copy import deepcopy

from langchaint import without_schema_titles


def test_without_schema_titles_distinguishes_annotations_from_names_and_data() -> None:
    literal = {"title": "Dune"}
    schema = {
        "title": "Book",
        "$defs": {"title": {"title": "Book title", "description": "Title on the cover"}},
        "properties": {"title": {"anyOf": [{"$ref": "#/$defs/title", "title": "Text"}, False]}},
        "required": ["title"],
        "items": {"title": "Item", "const": literal},
        "examples": [literal],
        "additionalItems": literal,
    }
    original = deepcopy(schema)
    assert without_schema_titles(schema) == {
        "$defs": {"title": {"description": "Title on the cover"}},
        "properties": {"title": {"anyOf": [{"$ref": "#/$defs/title"}, False]}},
        "required": ["title"],
        "items": {"const": {"title": "Dune"}},
        "examples": [{"title": "Dune"}],
        "additionalItems": {"title": "Dune"},
    }
    assert schema == original
