# `rdf_converter.py` (RDF 1.2 N-Triples <-> Turtle)

This README documents only `rdf_converter.py`.

`rdf_converter.py` is a self-contained Python parser/serializer/converter for:

- RDF 1.2 N-Triples (`.nt`)
- RDF 1.2 Turtle (`.ttl`, `.turtle`)

It does not depend on `rdflib` for parsing or conversion.

## Features

- Parse RDF 1.2 N-Triples into an internal typed triple model
- Parse RDF 1.2 Turtle (with base IRI support for relative IRIs)
- Convert `N-Triples -> Turtle`
- Convert `Turtle -> N-Triples`
- Deterministic serialization
- Optional readable Turtle output (grouping, object lists, prefix generation)
- Optional validation-only mode (parse without writing output)
- Optional graph/statistics reporting

## Requirements

- Python 3.10+ (uses modern type syntax, e.g. `X | Y`)

## CLI Usage

```bash
python3 rdf_converter.py INPUT [OUTPUT]
```

`INPUT` / `OUTPUT` can be file paths or `-` (stdin/stdout).

Examples:

```bash
# Convert by file extension
python3 rdf_converter.py input.nt output.ttl
python3 rdf_converter.py input.ttl output.nt

# Convert using stdin/stdout (explicit formats required)
python3 rdf_converter.py --from nt --to turtle - -

# Validate only (no output path required)
python3 rdf_converter.py --validate-only input.ttl
```

## Format Selection

Formats can be inferred from file extensions or passed explicitly.

Supported aliases:

- `nt`, `ntriples`, `n-triples`
- `ttl`, `turtle`

Use:

- `--from ...` for input format
- `--to ...` for output format

If inference is not possible (for example `-` / stdin), provide the format explicitly.

## Options

Core options:

- `--from {nt,ntriples,n-triples,ttl,turtle}`
- `--to {nt,ntriples,n-triples,ttl,turtle}`
- `--base BASE_IRI` (base IRI for Turtle input; defaults to input file URI when possible)
- `--validate-only` (parse only, do not write output)
- `--stats` (print graph/serialization stats to `stderr`)

Turtle output options (valid only when target format is Turtle):

- `--turtle-style {readable,minimal}`
- `--lists {auto,off}`
- `--auto-prefixes {on,off}`
- `--prefix PREFIX=IRI` (repeatable)
- `--output-base IRI` (emit `@base`)

Validation rules enforced by the CLI:

- `OUTPUT` is optional only with `--validate-only`
- source and target formats must differ when converting
- Turtle-only options are rejected for non-Turtle output
- conflicting repeated `--prefix` bindings are rejected

## Turtle Output Modes

`--turtle-style readable` (default):

- groups triples by subject
- groups repeated predicates/objects with `;` and `,`
- can compact RDF collections to Turtle lists (`(...)`)
- can emit `@prefix` directives automatically

`--turtle-style minimal`:

- writes one triple per line
- avoids readability-oriented compaction

## Examples

```bash
# Validate N-Triples and print stats
python3 rdf_converter.py --validate-only --stats input.nt

# Turtle output without list compaction
python3 rdf_converter.py --lists off input.nt output.ttl

# Minimal Turtle output
python3 rdf_converter.py --turtle-style minimal input.nt output.ttl

# Manual prefixes only
python3 rdf_converter.py --auto-prefixes off \
  --prefix ex=http://example.org/ \
  --prefix foaf=http://xmlns.com/foaf/0.1/ \
  input.nt output.ttl

# Emit @base in generated Turtle
python3 rdf_converter.py --output-base http://example.org/base/ input.nt output.ttl
```

## Python API (Library Use)

`rdf_converter.py` can also be imported as a module.

Main functions:

- `parse_ntriples(text, source="<string>") -> list[Triple]`
- `parse_turtle(text, source="<string>", base_iri=None) -> list[Triple]`
- `serialize_ntriples(triples) -> str`
- `serialize_turtle(triples, options=None) -> str`
- `serialize_turtle_with_meta(triples, options=None) -> tuple[str, dict[str, int]]`
- `convert_data(...) -> tuple[str, list[Triple], dict[str, int]]`

Common data types:

- `IRI`
- `BNode`
- `Literal`
- `TripleTerm`
- `ParseError`
- `TurtleSerializationOptions`

Example:

```python
from rdf_converter import parse_turtle, serialize_ntriples

ttl = '<http://example/s> <http://example/p> "x" .\\n'
triples = parse_turtle(ttl, source="example.ttl", base_iri="http://example/")
nt = serialize_ntriples(triples)
print(nt)
```

## Error Handling

- Syntax and deterministic parse failures raise `ParseError`
- CLI validation/configuration errors are reported as `Error: ...` and exit with status `1`

## Notes

- The implementation follows the repository grammars in `grammar/ntriples.bnf` and `grammar/turtle.bnf`
- Turtle parsing supports base IRI resolution for relative IRIs
- Serialization output is designed to remain parseable by the project parser

## License

This project is licensed under the MIT License.
