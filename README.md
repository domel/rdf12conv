# `rdf_converter.py` (RDF 1.2 N-Triples / N-Quads / Turtle / TriG)

This README documents only `rdf_converter.py`.

`rdf_converter.py` is a self-contained Python parser/serializer/converter for:

- RDF 1.2 N-Triples (`.nt`)
- RDF 1.2 N-Quads (`.nq`, `.nquads`)
- RDF 1.2 Turtle (`.ttl`, `.turtle`)
- RDF 1.2 TriG (`.trig`)

It does not depend on `rdflib` for parsing or conversion.

## Features

- Parse RDF 1.2 N-Triples and Turtle into a typed triple model
- Parse RDF 1.2 N-Quads and TriG into a typed quad/dataset model
- Convert across triple and dataset syntaxes (`nt`, `nq`, `turtle`, `trig`)
- Deterministic serialization (`N-Triples`, `N-Quads`)
- Optional readable Turtle/TriG output (grouping, object lists, list compaction, prefixes)
- Optional validation-only mode (parse without writing output)
- Optional graph/dataset/statistics reporting

## Requirements

- Python 3.10+ (uses modern type syntax, e.g. `X | Y`)

## Installation

From PyPI (after publishing):

```bash
pip install rdf12conv
```

CLI command after installation:

```bash
rdf12conv INPUT [OUTPUT]
```

## Docker

Build the image from the repository root:

```bash
docker build -t rdf12conv .
```

Run the CLI (defaults to `--help` if no arguments are provided):

```bash
docker run --rm rdf12conv
```

Process files from the current directory by mounting it into `/work` (the container working directory):

```bash
docker run --rm -v "$PWD:/work" rdf12conv input.ttl output.nq
```

Examples with options:

```bash
docker run --rm -v "$PWD:/work" rdf12conv --validate-only --stats input.trig
docker run --rm -v "$PWD:/work" rdf12conv --into-graph http://example.org/g input.nt output.trig
```

## CLI Usage

```bash
python3 rdf_converter.py INPUT [OUTPUT]
```

`INPUT` / `OUTPUT` can be file paths or `-` (stdin/stdout).

Examples:

```bash
# Convert by file extension
python3 rdf_converter.py input.nt output.ttl
python3 rdf_converter.py input.trig output.nq

# Convert using stdin/stdout (explicit formats required)
python3 rdf_converter.py --from nq --to trig - -

# Validate only (no output path required)
python3 rdf_converter.py --validate-only input.trig
```

## Format Selection

Formats can be inferred from file extensions or passed explicitly.

Supported aliases:

- `nt`, `ntriples`, `n-triples`
- `nq`, `nquads`, `n-quads`
- `ttl`, `turtle`
- `trig`

Use:

- `--from ...` for input format
- `--to ...` for output format

If inference is not possible (for example `-` / stdin), provide the format explicitly.

## Options

Core options:

- `--from {nt,ntriples,n-triples,nq,nquads,n-quads,ttl,turtle,trig}`
- `--to {nt,ntriples,n-triples,nq,nquads,n-quads,ttl,turtle,trig}`
- `--base BASE_IRI` (base IRI for Turtle/TriG input; defaults to input file URI when possible)
- `--dataset-to-graph-policy {strict,default-only,select,union}` (how to project `nq`/`trig` datasets to `nt`/`turtle`)
- `--graph GRAPH_LABEL` (graph label for `--dataset-to-graph-policy select`; absolute IRI or `_:label`)
- `--into-graph GRAPH_LABEL` (when converting `nt`/`turtle` to `nq`/`trig`, put all triples in one named graph)
- `--validate-only` (parse only, do not write output)
- `--stats` (print graph/dataset/serialization stats to `stderr`)

Turtle/TriG output options (valid only when target format is Turtle or TriG):

- `--turtle-style {readable,minimal}`
- `--lists {auto,off}`
- `--auto-prefixes {on,off}`
- `--prefix PREFIX=IRI` (repeatable)
- `--output-base IRI` (emit `@base`)

Validation rules enforced by the CLI:

- `OUTPUT` is optional only with `--validate-only`
- source and target formats must differ when converting
- Turtle/TriG-only options are rejected for non-Turtle/TriG output
- `--dataset-to-graph-policy` / `--graph` apply only to `nt` / `turtle` output
- `--into-graph` applies only to `nt`/`turtle` input with `nq`/`trig` output
- `strict` (default) rejects converting a dataset with named graphs to `nt`/`turtle`
- conflicting repeated `--prefix` bindings are rejected

## Turtle / TriG Output Modes

`--turtle-style readable` (default):

- groups triples by subject
- groups repeated predicates/objects with `;` and `,`
- can compact RDF collections to Turtle lists (`(...)`)
- can emit `@prefix` directives automatically

`--turtle-style minimal`:

- writes one statement per line (or per line inside TriG graph blocks)
- avoids readability-oriented compaction

## Examples

```bash
# Validate N-Triples and print stats
python3 rdf_converter.py --validate-only --stats input.nt

# Validate TriG and print dataset stats
python3 rdf_converter.py --validate-only --stats input.trig

# Turtle output without list compaction
python3 rdf_converter.py --lists off input.nt output.ttl

# Readable TriG output from N-Quads
python3 rdf_converter.py input.nq output.trig

# Put all triples into one named graph when up-converting graph -> dataset
python3 rdf_converter.py --into-graph http://example.org/g input.ttl output.trig

# Minimal TriG output
python3 rdf_converter.py --turtle-style minimal input.nq output.trig

# Manual prefixes only (Turtle/TriG)
python3 rdf_converter.py --auto-prefixes off \
  --prefix ex=http://example.org/ \
  --prefix foaf=http://xmlns.com/foaf/0.1/ \
  input.nq output.trig

# Emit @base in generated Turtle/TriG
python3 rdf_converter.py --output-base http://example.org/base/ input.nt output.ttl

# Lossy dataset -> graph conversion policies
python3 rdf_converter.py --dataset-to-graph-policy default-only input.trig output.ttl
python3 rdf_converter.py --dataset-to-graph-policy union input.nq output.nt
python3 rdf_converter.py --dataset-to-graph-policy select \
  --graph http://example.org/g \
  input.trig output.ttl
```

## Dataset -> Graph Conversion Policies

When converting `nq` / `trig` to `nt` / `turtle`, graph labels cannot be represented.
The CLI therefore requires an explicit policy if you want lossy behavior.

- `strict` (default): fail if any named graph statements are present
- `default-only`: keep only the default graph, drop all named graphs
- `select`: keep only statements from one selected graph (`--graph ...`)
- `union`: flatten all dataset statements (default + named graphs) into one graph

For `nt` / `turtle` -> `nq` / `trig`, `--into-graph` can place all input triples into a single named graph instead of the default graph.

## Python API (Library Use)

`rdf_converter.py` can also be imported as a module.

Main functions:

- `parse_ntriples(text, source="<string>") -> list[Triple]`
- `parse_nquads(text, source="<string>") -> list[Quad]`
- `parse_turtle(text, source="<string>", base_iri=None) -> list[Triple]`
- `parse_trig(text, source="<string>", base_iri=None) -> list[Quad]`
- `serialize_ntriples(triples) -> str`
- `serialize_nquads(quads) -> str`
- `serialize_turtle(triples, options=None) -> str`
- `serialize_turtle_with_meta(triples, options=None) -> tuple[str, dict[str, int]]`
- `serialize_trig(quads, options=None) -> str`
- `serialize_trig_with_meta(quads, options=None) -> tuple[str, dict[str, int]]`
- `convert_data(...) -> tuple[str, list[Quad], dict[str, int]]`

Common data types:

- `IRI`
- `BNode`
- `Literal`
- `TripleTerm`
- `GraphLabel`
- `Triple`
- `Quad`
- `ParseError`
- `TurtleSerializationOptions`

Example:

```python
from rdf_converter import parse_trig, serialize_nquads

trig = '<http://example/g> { <http://example/s> <http://example/p> "x" . }\\n'
quads = parse_trig(trig, source="example.trig", base_iri="http://example/")
nq = serialize_nquads(quads)
print(nq)
```

## Error Handling

- Syntax and deterministic parse failures raise `ParseError`
- CLI validation/configuration errors are reported as `Error: ...` and exit with status `1`

## Notes

- The implementation follows the repository grammars in `grammar/*.bnf`
- Turtle and TriG parsing support base IRI resolution for relative IRIs
- Serialization output is designed to remain parseable by the project parser

## License

This project is licensed under the MIT License.
