#!/usr/bin/env python3
"""RDF 1.2 converter between N-Triples, N-Quads, Turtle, and TriG.

Implementation is self-contained and does not rely on rdflib.
It follows repository grammars from ``grammar/*.bnf``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

__version__ = "0.2.0"

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"

RDF_TYPE_IRI = f"{RDF_NS}type"
RDF_FIRST_IRI = f"{RDF_NS}first"
RDF_REST_IRI = f"{RDF_NS}rest"
RDF_NIL_IRI = f"{RDF_NS}nil"
RDF_REIFIES_IRI = f"{RDF_NS}reifies"

XSD_BOOLEAN_IRI = f"{XSD_NS}boolean"
XSD_INTEGER_IRI = f"{XSD_NS}integer"
XSD_DECIMAL_IRI = f"{XSD_NS}decimal"
XSD_DOUBLE_IRI = f"{XSD_NS}double"

RDF_LANG_STRING_IRI = f"{RDF_NS}langString"
RDF_DIR_LANG_STRING_IRI = f"{RDF_NS}dirLangString"

NAME_PUNCTUATION = "_-"


class ParseError(ValueError):
    """Raised on deterministic syntax/semantic parse errors."""

    def __init__(self, source: str, line: int, column: int, message: str):
        """Initialize a parse error with source location details."""
        super().__init__(f"{source}:{line}:{column}: {message}")
        self.source = source
        self.line = line
        self.column = column
        self.message = message


@dataclass(frozen=True)
class IRI:
    """IRI node value used by the parser and serializers."""
    value: str


@dataclass(frozen=True)
class BNode:
    """Blank node identifier used in parsed RDF triples."""
    label: str


@dataclass(frozen=True)
class Literal:
    """RDF literal value with optional language, direction, or datatype."""
    value: str
    lang: str | None = None
    direction: str | None = None
    datatype: str | None = None


@dataclass(frozen=True)
class TripleTerm:
    """Quoted triple term used as an RDF-star style node value."""
    subject: IRI | BNode
    predicate: IRI
    object: IRI | BNode | Literal | TripleTerm


Node = IRI | BNode | Literal | TripleTerm
Triple = tuple[IRI | BNode, IRI, Node]
GraphLabel = IRI | BNode
Quad = tuple[IRI | BNode, IRI, Node, GraphLabel | None]


@dataclass
class AnnotationData:
    """Stores parsed annotation/reification information for a statement."""
    reifiers: list[IRI | BNode] = field(default_factory=list)
    blocks: list[list[tuple[IRI, list[tuple[Node, "AnnotationData"]]]]] = field(
        default_factory=list
    )
    events: list[tuple[str, object]] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        """Return whether any annotation payload was collected."""
        return bool(self.events or self.reifiers or self.blocks)


class Scanner:
    """Stateful character scanner with line and column tracking."""
    def __init__(self, text: str, source: str):
        """Initialize scanner state for the provided source text."""
        self.text = text
        self.source = source
        self.i = 0
        self.line = 1
        self.col = 1

    def eof(self) -> bool:
        """Return `True` when the scanner reached the end of input."""
        return self.i >= len(self.text)

    def peek(self, offset: int = 0) -> str:
        """Return the character at the current position plus an optional offset."""
        idx = self.i + offset
        if idx >= len(self.text):
            return ""
        return self.text[idx]

    def startswith(self, token: str) -> bool:
        """Return `True` if the remaining input starts with `token`."""
        return self.text.startswith(token, self.i)

    def advance(self) -> str:
        """Consume and return one character while updating line/column counters."""
        if self.eof():
            self.error("unexpected end of input")
        ch = self.text[self.i]
        self.i += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def consume(self, token: str) -> bool:
        """Consume `token` if present and return whether it matched."""
        if not self.startswith(token):
            return False
        for _ in token:
            self.advance()
        return True

    def expect(self, token: str, message: str | None = None) -> None:
        """Consume `token` or raise a parse error with a helpful message."""
        if not self.consume(token):
            self.error(message or f"expected '{token}'")

    def mark(self) -> tuple[int, int, int]:
        """Capture the current scanner position for backtracking."""
        return self.i, self.line, self.col

    def reset(self, mark: tuple[int, int, int]) -> None:
        """Restore a position previously returned by `mark()`."""
        self.i, self.line, self.col = mark

    def error(self, message: str) -> None:
        """Raise `ParseError` at the current scanner position."""
        raise ParseError(self.source, self.line, self.col, message)


def is_space(ch: str) -> bool:
    """Return whether a character is RDF whitespace."""
    return ch in " \t\r\n"


def skip_ws_comments(scanner: Scanner) -> None:
    """Skip whitespace and `#` comments in the current scanner."""
    while not scanner.eof():
        ch = scanner.peek()
        if is_space(ch):
            scanner.advance()
            continue
        if ch == "#":
            while not scanner.eof() and scanner.peek() not in "\r\n":
                scanner.advance()
            continue
        break


def _in_ranges(cp: int, ranges: Iterable[tuple[int, int]]) -> bool:
    """Internal helper for in ranges."""
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False


PN_BASE_RANGES = (
    (0x00C0, 0x00D6),
    (0x00D8, 0x00F6),
    (0x00F8, 0x02FF),
    (0x0370, 0x037D),
    (0x037F, 0x1FFF),
    (0x200C, 0x200D),
    (0x2070, 0x218F),
    (0x2C00, 0x2FEF),
    (0x3001, 0xD7FF),
    (0xF900, 0xFDCF),
    (0xFDF0, 0xFFFD),
    (0x10000, 0xEFFFF),
)


def is_pn_chars_base(ch: str) -> bool:
    """Return whether a character is a valid `PN_CHARS_BASE` code point."""
    if len(ch) != 1:
        return False
    if "A" <= ch <= "Z" or "a" <= ch <= "z":
        return True
    cp = ord(ch)
    return _in_ranges(cp, PN_BASE_RANGES)


def is_pn_chars_u(ch: str) -> bool:
    """Return whether a character is a valid `PN_CHARS_U` code point."""
    return ch == "_" or is_pn_chars_base(ch)


def is_pn_chars(ch: str) -> bool:
    """Return whether a character is a valid `PN_CHARS` code point."""
    if len(ch) != 1:
        return False
    if is_pn_chars_u(ch) or ch in "-0123456789":
        return True
    cp = ord(ch)
    return cp == 0x00B7 or 0x0300 <= cp <= 0x036F or 0x203F <= cp <= 0x2040


def is_hex(ch: str) -> bool:
    """Return whether a character is a hexadecimal digit."""
    return ch.isdigit() or ("A" <= ch <= "F") or ("a" <= ch <= "f")


def decode_uchar(scanner: Scanner) -> str:
    """Decode Unicode escapes."""
    def read_hex4() -> int:
        """Read four hexadecimal digits and return their integer value."""
        digits = []
        for _ in range(4):
            ch = scanner.peek()
            if not is_hex(ch):
                scanner.error("invalid \\u escape")
            digits.append(scanner.advance())
        return int("".join(digits), 16)

    if scanner.consume("\\u"):
        codepoint = read_hex4()
        if 0xD800 <= codepoint <= 0xDBFF:
            # Surrogate pair is allowed only when followed by another \\u low surrogate.
            if not scanner.consume("\\u"):
                scanner.error("high surrogate must be followed by low surrogate")
            low = read_hex4()
            if not (0xDC00 <= low <= 0xDFFF):
                scanner.error("invalid low surrogate in pair")
            scalar = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
            return chr(scalar)
        if 0xDC00 <= codepoint <= 0xDFFF:
            scanner.error("lone low surrogate is not allowed")
        return chr(codepoint)
    if scanner.consume("\\U"):
        digits = []
        for _ in range(8):
            ch = scanner.peek()
            if not is_hex(ch):
                scanner.error("invalid \\U escape")
            digits.append(scanner.advance())
        codepoint = int("".join(digits), 16)
        if codepoint > 0x10FFFF:
            scanner.error("code point out of range")
        if 0xD800 <= codepoint <= 0xDFFF:
            scanner.error("surrogate code points are not allowed")
        return chr(codepoint)
    scanner.error("expected unicode escape")


def decode_echar(scanner: Scanner, allow_single_quote: bool) -> str:
    """Decode ECHAR escapes."""
    scanner.expect("\\", "expected escape")
    ch = scanner.peek()
    mapping = {
        "t": "\t",
        "b": "\b",
        "n": "\n",
        "r": "\r",
        "f": "\f",
        '"': '"',
        "\\": "\\",
    }
    if allow_single_quote:
        mapping["'"] = "'"
    if ch not in mapping:
        scanner.error("invalid escape sequence")
    scanner.advance()
    return mapping[ch]


def parse_iri_ref(scanner: Scanner, require_absolute: bool, allow_empty: bool) -> str:
    """Parse IRI reference from the current input and return the result."""
    scanner.expect("<")
    chars: list[str] = []
    while True:
        if scanner.eof():
            scanner.error("unterminated IRI")
        ch = scanner.peek()
        if ch == ">":
            scanner.advance()
            iri = "".join(chars)
            validate_iri(
                iri, require_absolute=require_absolute, allow_empty=allow_empty
            )
            return iri
        if ch == "\\":
            uch = decode_uchar(scanner)
            if uch in '<>"{}|^`\\' or ord(uch) <= 0x20:
                scanner.error("invalid escaped character in IRI")
            chars.append(uch)
            continue
        if ch in '<>"{}|^`':
            scanner.error("invalid character in IRI")
        if ord(ch) <= 0x20:
            scanner.error("invalid whitespace/control in IRI")
        chars.append(scanner.advance())


def validate_iri(value: str, require_absolute: bool, allow_empty: bool) -> None:
    """Validate IRI."""
    if not value and not allow_empty:
        raise ValueError("IRI must not be empty")
    if any(ord(ch) <= 0x20 for ch in value):
        raise ValueError("IRI contains whitespace/control")
    parts = urlsplit(value)
    if require_absolute and not parts.scheme:
        raise ValueError("IRI must be absolute")


def encode_iri_ref(value: str) -> str:
    """Encode IRI reference."""
    out: list[str] = ["<"]
    for ch in value:
        cp = ord(ch)
        if ch in '<>"{}|^`\\' or cp <= 0x20:
            if cp <= 0xFFFF:
                out.append(f"\\u{cp:04X}")
            else:
                out.append(f"\\U{cp:08X}")
        else:
            out.append(ch)
    out.append(">")
    return "".join(out)


def _remove_dot_segments(path: str) -> str:
    """Normalize a path by removing `.` and `..` dot segments."""
    input_buffer = path
    output_buffer = ""

    def remove_last_segment(buf: str) -> str:
        """Drop the final path segment while preserving leading slash semantics."""
        idx = buf.rfind("/")
        if idx < 0:
            return ""
        return buf[:idx]

    while input_buffer:
        if input_buffer.startswith("../"):
            input_buffer = input_buffer[3:]
            continue
        if input_buffer.startswith("./"):
            input_buffer = input_buffer[2:]
            continue
        if input_buffer.startswith("/./"):
            input_buffer = "/" + input_buffer[3:]
            continue
        if input_buffer == "/.":
            input_buffer = "/"
            continue
        if input_buffer.startswith("/../"):
            input_buffer = "/" + input_buffer[4:]
            output_buffer = remove_last_segment(output_buffer)
            continue
        if input_buffer == "/..":
            input_buffer = "/"
            output_buffer = remove_last_segment(output_buffer)
            continue
        if input_buffer == "." or input_buffer == "..":
            input_buffer = ""
            continue

        if input_buffer.startswith("/"):
            next_slash = input_buffer.find("/", 1)
        else:
            next_slash = input_buffer.find("/")
        if next_slash < 0:
            segment = input_buffer
            input_buffer = ""
        else:
            segment = input_buffer[:next_slash]
            input_buffer = input_buffer[next_slash:]
        output_buffer += segment

    return output_buffer


def _merge_reference_path(base_path: str, base_has_authority: bool, ref_path: str) -> str:
    """Merge an RFC 3986 reference path against the base path."""
    if base_has_authority and base_path == "":
        return "/" + ref_path
    slash = base_path.rfind("/")
    if slash < 0:
        return ref_path
    return base_path[: slash + 1] + ref_path


def resolve_iri_reference(base_iri: str, ref_iri: str) -> str:
    """Resolve a relative IRI reference against a base without collapsing // path segments."""

    ref_parts = urlsplit(ref_iri)
    if ref_parts.scheme:
        resolved = urlunsplit(
            (
                ref_parts.scheme,
                ref_parts.netloc,
                _remove_dot_segments(ref_parts.path),
                ref_parts.query,
                ref_parts.fragment,
            )
        )
    else:
        base_parts = urlsplit(base_iri)
        if ref_parts.netloc:
            path = _remove_dot_segments(ref_parts.path)
            resolved = urlunsplit(
                (base_parts.scheme, ref_parts.netloc, path, ref_parts.query, ref_parts.fragment)
            )
        else:
            if ref_parts.path == "":
                path = base_parts.path
                query = ref_parts.query if "?" in ref_iri.split("#", 1)[0] else base_parts.query
            elif ref_parts.path.startswith("/"):
                path = _remove_dot_segments(ref_parts.path)
                query = ref_parts.query
            else:
                merged = _merge_reference_path(
                    base_parts.path,
                    base_has_authority=bool(base_parts.netloc),
                    ref_path=ref_parts.path,
                )
                path = _remove_dot_segments(merged)
                query = ref_parts.query
            resolved = urlunsplit(
                (base_parts.scheme, base_parts.netloc, path, query, ref_parts.fragment)
            )

    # Preserve explicit empty query/fragment markers (urlunsplit drops them).
    ref_no_fragment, _, _ = ref_iri.partition("#")
    has_query_marker = "?" in ref_no_fragment
    has_fragment_marker = "#" in ref_iri
    if has_query_marker and ref_parts.query == "":
        head, sep, tail = resolved.partition("#")
        if "?" not in head:
            resolved = f"{head}?{sep}{tail}" if sep else f"{head}?"
    if has_fragment_marker and ref_parts.fragment == "" and "#" not in resolved:
        resolved += "#"
    return resolved


def parse_short_string(
    scanner: Scanner, quote: str, allow_single_quote_escape: bool
) -> str:
    """Parse short string from the current input and return the result."""
    scanner.expect(quote)
    out: list[str] = []
    while True:
        if scanner.eof():
            scanner.error("unterminated string")
        ch = scanner.peek()
        if ch == quote:
            scanner.advance()
            return "".join(out)
        if ch in "\n\r":
            scanner.error("newline in short string literal")
        if ch == "\\":
            mark = scanner.mark()
            scanner.advance()
            esc = scanner.peek()
            scanner.reset(mark)
            if esc in "uU":
                out.append(decode_uchar(scanner))
            else:
                out.append(
                    decode_echar(scanner, allow_single_quote=allow_single_quote_escape)
                )
            continue
        out.append(scanner.advance())


def parse_long_string(
    scanner: Scanner, quote: str, allow_single_quote_escape: bool
) -> str:
    """Parse long string from the current input and return the result."""
    delim = quote * 3
    scanner.expect(delim)
    out: list[str] = []
    while True:
        if scanner.eof():
            scanner.error("unterminated long string")
        if scanner.startswith(delim):
            scanner.consume(delim)
            return "".join(out)
        ch = scanner.peek()
        if ch == "\\":
            mark = scanner.mark()
            scanner.advance()
            esc = scanner.peek()
            scanner.reset(mark)
            if esc in "uU":
                out.append(decode_uchar(scanner))
            else:
                out.append(
                    decode_echar(scanner, allow_single_quote=allow_single_quote_escape)
                )
            continue
        out.append(scanner.advance())


def parse_lang_dir(scanner: Scanner) -> tuple[str, str | None]:
    """Parse language and direction suffix from the current input and return the result."""
    scanner.expect("@")
    primary = []
    while scanner.peek().isalpha():
        primary.append(scanner.advance())
    if not primary:
        scanner.error("invalid language tag")
    if len(primary) > 8:
        scanner.error("language subtag too long")
    subtags: list[str] = []
    while scanner.peek() == "-" and scanner.peek(1) != "-":
        scanner.advance()
        segment = []
        while scanner.peek().isalnum():
            segment.append(scanner.advance())
        if not segment:
            scanner.error("empty language subtag")
        if len(segment) > 8:
            scanner.error("language subtag too long")
        subtags.append("".join(segment))
    lang = "".join(primary).lower()
    if subtags:
        lang += "-" + "-".join(segment.lower() for segment in subtags)
    direction = None
    if scanner.consume("--"):
        letters = []
        while scanner.peek().isalpha():
            letters.append(scanner.advance())
        if not letters:
            scanner.error("missing text direction")
        direction = "".join(letters)
        if direction not in ("ltr", "rtl"):
            scanner.error("text direction must be 'ltr' or 'rtl'")
    return lang, direction


def escape_string_value(value: str) -> str:
    """Escape string value."""
    out: list[str] = []
    for ch in value:
        cp = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\f":
            out.append("\\f")
        elif cp < 0x20 or cp in (0x7F, 0xFFFE, 0xFFFF):
            out.append(f"\\u{cp:04X}")
        else:
            out.append(ch)
    return "".join(out)


def is_name_boundary(ch: str) -> bool:
    """Return whether a character terminates a Turtle/N-Triples keyword token."""
    if not ch:
        return True
    return is_space(ch) or ch in ";,.()[]{}<>\"'|#"


def is_keyword(scanner: Scanner, kw: str) -> bool:
    """Return whether the scanner is positioned at the exact keyword."""
    if not scanner.startswith(kw):
        return False
    nxt = scanner.peek(len(kw))
    return is_name_boundary(nxt)


def is_keyword_ci(scanner: Scanner, kw: str) -> bool:
    """Return whether the scanner matches a keyword case-insensitively."""
    end = scanner.i + len(kw)
    if end > len(scanner.text):
        return False
    if scanner.text[scanner.i : end].lower() != kw.lower():
        return False
    nxt = scanner.peek(len(kw))
    return is_name_boundary(nxt)


def is_keyword_with_extra_boundary(scanner: Scanner, kw: str, extra: str) -> bool:
    """Return whether keyword matches and is followed by a standard or extra boundary."""
    if not scanner.startswith(kw):
        return False
    nxt = scanner.peek(len(kw))
    return is_name_boundary(nxt) or nxt in extra


def is_keyword_ci_with_extra_boundary(scanner: Scanner, kw: str, extra: str) -> bool:
    """Case-insensitive keyword match allowing additional boundary characters."""
    end = scanner.i + len(kw)
    if end > len(scanner.text):
        return False
    if scanner.text[scanner.i : end].lower() != kw.lower():
        return False
    nxt = scanner.peek(len(kw))
    return is_name_boundary(nxt) or nxt in extra


class BaseParser:
    """Shared parser utilities used by the N-Triples and Turtle parsers."""
    def __init__(self, text: str, source: str):
        """Initialize the `BaseParser` instance."""
        self.scanner = Scanner(text, source)
        self.triples: list[Triple] = []
        self._generated_bnode = 0
        self._reserved_labels: set[str] = set()

    def new_bnode(self) -> BNode:
        """Create a fresh generated blank node label that avoids collisions."""
        while True:
            label = f"genid{self._generated_bnode}"
            self._generated_bnode += 1
            if label not in self._reserved_labels:
                self._reserved_labels.add(label)
                return BNode(label)

    def emit(self, subject: IRI | BNode, predicate: IRI, obj: Node) -> None:
        """Append one triple to the parser output buffer."""
        self.triples.append((subject, predicate, obj))

    def emit_annotation(
        self,
        subject: IRI | BNode,
        predicate: IRI,
        obj: Node,
        annotation: AnnotationData,
    ) -> None:
        """Emit triples representing parsed statement annotations/reification."""
        if not annotation.has_data:
            return
        triple_term = TripleTerm(subject, predicate, obj)
        if annotation.events:
            current_subject: IRI | BNode | None = None
            for kind, payload in annotation.events:
                if kind == "reifier":
                    current_subject = payload  # type: ignore[assignment]
                    self.emit(current_subject, IRI(RDF_REIFIES_IRI), triple_term)
                    continue
                if kind == "block":
                    if current_subject is None:
                        current_subject = self.new_bnode()
                        self.emit(current_subject, IRI(RDF_REIFIES_IRI), triple_term)
                    self.emit_pairs_from_structure(
                        current_subject,
                        payload,  # type: ignore[arg-type]
                    )
                    # A block applies to a single reifier occurrence in sequence.
                    current_subject = None
                    continue
                raise TypeError(f"unsupported annotation event kind: {kind!r}")
            return
        ann_subject: IRI | BNode
        if annotation.reifiers:
            ann_subject = annotation.reifiers[0]
            for extra in annotation.reifiers[1:]:
                self.emit(extra, IRI(RDF_REIFIES_IRI), triple_term)
        else:
            ann_subject = self.new_bnode()
        self.emit(ann_subject, IRI(RDF_REIFIES_IRI), triple_term)
        for block in annotation.blocks:
            self.emit_pairs_from_structure(ann_subject, block)

    def emit_pairs_from_structure(
        self,
        subject: IRI | BNode,
        pairs: list[tuple[IRI, list[tuple[Node, AnnotationData]]]],
    ) -> None:
        """Emit predicate-object pairs from a parsed block/list structure."""
        for predicate, objects in pairs:
            for obj, annotation in objects:
                self.emit(subject, predicate, obj)
                self.emit_annotation(subject, predicate, obj, annotation)


class NTriplesParser(BaseParser):
    """Parser for the repository's RDF 1.2 N-Triples grammar."""
    def parse(self) -> list[Triple]:
        """Parse the current document and return parsed triples."""
        while True:
            skip_ws_comments(self.scanner)
            if self.scanner.eof():
                return self.triples
            if is_keyword(self.scanner, "VERSION"):
                self.parse_version_directive()
                continue
            self.parse_statement()

    def parse_version_directive(self) -> None:
        """Parse version directive from the current input and return the result."""
        self.scanner.expect("VERSION")
        skip_ws_comments(self.scanner)
        # Grammar allows string literal only.
        _ = parse_short_string(self.scanner, '"', allow_single_quote_escape=False)

    def parse_statement(self) -> None:
        """Parse statement from the current input and return the result."""
        subject = self.parse_subject()
        skip_ws_comments(self.scanner)
        predicate = self.parse_predicate()
        skip_ws_comments(self.scanner)
        obj = self.parse_object()

        annotation = self.parse_annotation_data()
        if annotation.has_data and isinstance(subject, BNode):
            self.scanner.error(
                "annotated N-Triples statements cannot have blank node subjects"
            )
        if annotation.has_data and isinstance(obj, BNode):
            self.scanner.error(
                "annotated N-Triples statements cannot have blank node objects"
            )

        skip_ws_comments(self.scanner)
        self.scanner.expect(".", "expected '.' to end N-Triples statement")

        self.emit(subject, predicate, obj)
        self.emit_annotation(subject, predicate, obj, annotation)

    def parse_annotation_data(self) -> AnnotationData:
        """Parse annotation data from the current input and return the result."""
        annotation = AnnotationData()
        while True:
            skip_ws_comments(self.scanner)
            if not self.scanner.startswith("{|"):
                break
            block = self.parse_annotation_block_structure()
            annotation.blocks.append(block)
            annotation.events.append(("block", block))
        return annotation

    def parse_annotation_block_structure(
        self,
    ) -> list[tuple[IRI, list[tuple[Node, AnnotationData]]]]:
        """Parse annotation block structure from the current input and return the result."""
        self.scanner.expect("{|")
        skip_ws_comments(self.scanner)
        pairs = self.parse_pairs_structure(terminator="|}", allow_a_verb=False)
        skip_ws_comments(self.scanner)
        self.scanner.expect("|}", "expected '|}' to close annotation block")
        return pairs

    def parse_pairs_structure(
        self,
        terminator: str,
        allow_a_verb: bool,
    ) -> list[tuple[IRI, list[tuple[Node, AnnotationData]]]]:
        """Parse pairs structure from the current input and return the result."""
        pairs: list[tuple[IRI, list[tuple[Node, AnnotationData]]]] = []
        while True:
            if self.scanner.startswith(terminator):
                break
            pred = self.parse_verb(allow_a=allow_a_verb)
            skip_ws_comments(self.scanner)
            objs = self.parse_object_list_structure()
            pairs.append((pred, objs))
            skip_ws_comments(self.scanner)
            if not self.scanner.consume(";"):
                break
            skip_ws_comments(self.scanner)
            while self.scanner.consume(";"):
                skip_ws_comments(self.scanner)
            if self.scanner.startswith(terminator):
                break
        return pairs

    def parse_object_list_structure(self) -> list[tuple[Node, AnnotationData]]:
        """Parse object list structure from the current input and return the result."""
        objects: list[tuple[Node, AnnotationData]] = []
        while True:
            obj = self.parse_object()
            annotation = self.parse_annotation_data()
            objects.append((obj, annotation))
            skip_ws_comments(self.scanner)
            if not self.scanner.consume(","):
                break
            skip_ws_comments(self.scanner)
        return objects

    def parse_subject(self) -> IRI | BNode:
        """Parse subject from the current input and return the result."""
        ch = self.scanner.peek()
        if ch == "<":
            return IRI(self.parse_iri(require_absolute=True))
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        self.scanner.error("N-Triples subject must be IRI or blank node")

    def parse_predicate(self) -> IRI:
        """Parse predicate from the current input and return the result."""
        if self.scanner.peek() != "<":
            self.scanner.error("N-Triples predicate must be IRI")
        return IRI(self.parse_iri(require_absolute=True))

    def parse_verb(self, allow_a: bool) -> IRI:
        """Parse verb from the current input and return the result."""
        if allow_a and is_keyword(self.scanner, "a"):
            self.scanner.expect("a")
            return IRI(RDF_TYPE_IRI)
        return self.parse_predicate()

    def parse_object(self) -> Node:
        """Parse object from the current input and return the result."""
        if self.scanner.startswith("<<("):
            return self.parse_triple_term()
        if self.scanner.startswith("<<"):
            self.scanner.error("invalid RDF-star syntax, expected '<<( ... )>>'")
        ch = self.scanner.peek()
        if ch == "<":
            return IRI(self.parse_iri(require_absolute=True))
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        if ch == '"':
            return self.parse_literal()
        self.scanner.error("invalid N-Triples object")

    def parse_literal(self) -> Literal:
        """Parse literal from the current input and return the result."""
        value = parse_short_string(self.scanner, '"', allow_single_quote_escape=False)
        lang = None
        direction = None
        datatype = None
        while is_space(self.scanner.peek()):
            self.scanner.advance()
        if self.scanner.consume("^^"):
            while is_space(self.scanner.peek()):
                self.scanner.advance()
            datatype = self.parse_iri(require_absolute=True)
            if datatype in (RDF_LANG_STRING_IRI, RDF_DIR_LANG_STRING_IRI):
                self.scanner.error(
                    "rdf:langString and rdf:dirLangString datatypes are not allowed"
                )
        elif self.scanner.peek() == "@":
            lang, direction = parse_lang_dir(self.scanner)
        return Literal(value=value, lang=lang, direction=direction, datatype=datatype)

    def parse_blank_node_label(self) -> BNode:
        """Parse blank node label from the current input and return the result."""
        self.scanner.expect("_:")
        if not (is_pn_chars_u(self.scanner.peek()) or self.scanner.peek().isdigit()):
            self.scanner.error("invalid blank node label")
        chars = [self.scanner.advance()]
        while True:
            ch = self.scanner.peek()
            if not ch:
                break
            if is_pn_chars(ch):
                chars.append(self.scanner.advance())
                continue
            if ch == ".":
                nxt = self.scanner.peek(1)
                if nxt and (nxt == "." or is_pn_chars(nxt)):
                    chars.append(self.scanner.advance())
                    continue
                break
            break
        label = "".join(chars)
        self._reserved_labels.add(label)
        return BNode(label)

    def parse_iri(self, require_absolute: bool) -> str:
        """Parse IRI from the current input and return the result."""
        try:
            return parse_iri_ref(
                self.scanner, require_absolute=require_absolute, allow_empty=False
            )
        except ValueError as exc:
            self.scanner.error(str(exc))
        raise AssertionError("unreachable")

    def parse_triple_term(self) -> TripleTerm:
        """Parse triple-term node from the current input and return the result."""
        self.scanner.expect("<<(")
        skip_ws_comments(self.scanner)
        subject = self.parse_tt_subject()
        skip_ws_comments(self.scanner)
        predicate = self.parse_predicate()
        skip_ws_comments(self.scanner)
        obj = self.parse_tt_object()
        skip_ws_comments(self.scanner)
        self.scanner.expect(")>>", "expected ')>>' to close triple term")
        return TripleTerm(subject=subject, predicate=predicate, object=obj)

    def parse_tt_subject(self) -> IRI | BNode:
        """Parse triple-term subject from the current input and return the result."""
        ch = self.scanner.peek()
        if ch == "<":
            return IRI(self.parse_iri(require_absolute=True))
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        self.scanner.error("triple term subject must be IRI or blank node")

    def parse_tt_object(self) -> Node:
        """Parse triple-term object from the current input and return the result."""
        if self.scanner.startswith("<<("):
            return self.parse_triple_term()
        ch = self.scanner.peek()
        if ch == "<":
            return IRI(self.parse_iri(require_absolute=True))
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        if ch == '"':
            return self.parse_literal()
        self.scanner.error("invalid triple term object")


class NQuadsParser(NTriplesParser):
    """Parser for the repository's RDF 1.2 N-Quads grammar."""

    def __init__(self, text: str, source: str):
        """Initialize the `NQuadsParser` instance."""
        super().__init__(text, source)
        self.quads: list[Quad] = []
        self._current_graph: GraphLabel | None = None

    def emit(self, subject: IRI | BNode, predicate: IRI, obj: Node) -> None:
        """Append one quad to the parser output buffer."""
        self.quads.append((subject, predicate, obj, self._current_graph))

    def parse(self) -> list[Quad]:
        """Parse the current document and return parsed quads."""
        while True:
            skip_ws_comments(self.scanner)
            if self.scanner.eof():
                return self.quads
            if is_keyword(self.scanner, "VERSION"):
                self.parse_version_directive()
                continue
            self.parse_quad_statement()

    def parse_quad_statement(self) -> None:
        """Parse one N-Quads statement."""
        subject = self.parse_subject()
        skip_ws_comments(self.scanner)
        predicate = self.parse_predicate()
        skip_ws_comments(self.scanner)
        obj = self.parse_object()

        skip_ws_comments(self.scanner)
        if self.scanner.startswith("{|"):
            self.scanner.error("annotation syntax is not permitted in N-Quads")
        graph: GraphLabel | None = None
        if not self.scanner.startswith("."):
            graph = self.parse_graph_label()
            skip_ws_comments(self.scanner)
        self.scanner.expect(".", "expected '.' to end N-Quads statement")

        previous_graph = self._current_graph
        self._current_graph = graph
        try:
            self.emit(subject, predicate, obj)
        finally:
            self._current_graph = previous_graph

    def parse_graph_label(self) -> GraphLabel:
        """Parse optional graph label position in an N-Quads statement."""
        if self.scanner.peek() == "<":
            return IRI(self.parse_iri(require_absolute=True))
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        self.scanner.error("N-Quads graph label must be IRI or blank node")


class TurtleParser(BaseParser):
    """Parser for the repository's RDF 1.2 Turtle grammar."""
    def __init__(self, text: str, source: str, base_iri: str | None):
        """Initialize the `TurtleParser` instance."""
        super().__init__(text, source)
        self.base_iri = base_iri
        self.prefixes: dict[str, str] = {}

    def parse(self) -> list[Triple]:
        """Parse the current document and return parsed triples."""
        while True:
            skip_ws_comments(self.scanner)
            if self.scanner.eof():
                return self.triples
            if self.parse_directive_if_present():
                continue
            self.parse_triples_statement()
            skip_ws_comments(self.scanner)
            self.scanner.expect(".", "expected '.' to end Turtle triple statement")

    def parse_directive_if_present(self) -> bool:
        """Parse directive if present from the current input and return the result."""
        if is_keyword_with_extra_boundary(self.scanner, "@prefix", ":"):
            self.scanner.expect("@prefix")
            skip_ws_comments(self.scanner)
            prefix = self.parse_pname_ns()
            skip_ws_comments(self.scanner)
            iri = self.parse_iri_ref_turtle()
            skip_ws_comments(self.scanner)
            self.scanner.expect(".", "expected '.' after @prefix")
            self.prefixes[prefix] = iri
            return True
        if is_keyword_with_extra_boundary(self.scanner, "@base", "<"):
            self.scanner.expect("@base")
            skip_ws_comments(self.scanner)
            self.base_iri = self.parse_iri_ref_turtle()
            skip_ws_comments(self.scanner)
            self.scanner.expect(".", "expected '.' after @base")
            return True
        if is_keyword_with_extra_boundary(self.scanner, "@version", "\"'"):
            self.scanner.expect("@version")
            skip_ws_comments(self.scanner)
            self.parse_version_specifier()
            skip_ws_comments(self.scanner)
            self.scanner.expect(".", "expected '.' after @version")
            return True
        if is_keyword_ci_with_extra_boundary(self.scanner, "PREFIX", ":"):
            for _ in "PREFIX":
                self.scanner.advance()
            skip_ws_comments(self.scanner)
            prefix = self.parse_pname_ns()
            skip_ws_comments(self.scanner)
            iri = self.parse_iri_ref_turtle()
            self.prefixes[prefix] = iri
            return True
        if is_keyword_ci_with_extra_boundary(self.scanner, "BASE", "<"):
            for _ in "BASE":
                self.scanner.advance()
            skip_ws_comments(self.scanner)
            self.base_iri = self.parse_iri_ref_turtle()
            return True
        if is_keyword_ci_with_extra_boundary(self.scanner, "VERSION", "\"'"):
            for _ in "VERSION":
                self.scanner.advance()
            skip_ws_comments(self.scanner)
            self.parse_version_specifier()
            return True
        return False

    def parse_version_specifier(self) -> str:
        """Parse version specifier from the current input and return the result."""
        if self.scanner.peek() == '"':
            return parse_short_string(self.scanner, '"', allow_single_quote_escape=True)
        if self.scanner.peek() == "'":
            return parse_short_string(self.scanner, "'", allow_single_quote_escape=True)
        self.scanner.error("expected version specifier string")

    def parse_triples_statement(self) -> None:
        """Parse triples statement from the current input and return the result."""
        skip_ws_comments(self.scanner)
        if self.scanner.startswith("<<") and not self.scanner.startswith("<<("):
            subject, _ = self.parse_reified_triple(needs_subject_reference=True)
            skip_ws_comments(self.scanner)
            if self.can_start_verb():
                pairs = self.parse_pairs_structure(
                    terminators=(".",), allow_a_verb=True
                )
                self.emit_pairs_from_structure(subject, pairs)
            return

        if self.scanner.peek() == "[":
            subject = self.parse_blank_node_property_list()
            skip_ws_comments(self.scanner)
            if self.can_start_verb():
                pairs = self.parse_pairs_structure(
                    terminators=(".",), allow_a_verb=True
                )
                self.emit_pairs_from_structure(subject, pairs)
            return

        subject = self.parse_subject()
        skip_ws_comments(self.scanner)
        pairs = self.parse_pairs_structure(terminators=(".",), allow_a_verb=True)
        self.emit_pairs_from_structure(subject, pairs)

    def parse_pairs_structure(
        self,
        terminators: tuple[str, ...],
        allow_a_verb: bool,
    ) -> list[tuple[IRI, list[tuple[Node, AnnotationData]]]]:
        """Parse pairs structure from the current input and return the result."""
        pairs: list[tuple[IRI, list[tuple[Node, AnnotationData]]]] = []
        predicate = self.parse_verb(allow_a=allow_a_verb)
        skip_ws_comments(self.scanner)
        objects = self.parse_object_list_structure()
        pairs.append((predicate, objects))

        while True:
            skip_ws_comments(self.scanner)
            if any(self.scanner.startswith(tok) for tok in terminators):
                break
            if not self.scanner.consume(";"):
                break
            skip_ws_comments(self.scanner)
            while self.scanner.consume(";"):
                skip_ws_comments(self.scanner)
            if any(self.scanner.startswith(tok) for tok in terminators):
                break
            if not self.can_start_verb():
                self.scanner.error("expected predicate after ';'")
            predicate = self.parse_verb(allow_a=allow_a_verb)
            skip_ws_comments(self.scanner)
            objects = self.parse_object_list_structure()
            pairs.append((predicate, objects))
        return pairs

    def parse_object_list_structure(self) -> list[tuple[Node, AnnotationData]]:
        """Parse object list structure from the current input and return the result."""
        out: list[tuple[Node, AnnotationData]] = []
        while True:
            obj = self.parse_object()
            annotation = self.parse_annotation_data()
            out.append((obj, annotation))
            skip_ws_comments(self.scanner)
            if not self.scanner.consume(","):
                break
            skip_ws_comments(self.scanner)
        return out

    def parse_annotation_data(self) -> AnnotationData:
        """Parse annotation data from the current input and return the result."""
        annotation = AnnotationData()
        while True:
            skip_ws_comments(self.scanner)
            if self.scanner.consume("~"):
                skip_ws_comments(self.scanner)
                if self.can_start_iri_or_blanknode():
                    reifier = self.parse_iri_or_blanknode()
                else:
                    reifier = self.new_bnode()
                annotation.reifiers.append(reifier)
                annotation.events.append(("reifier", reifier))
                continue
            if self.scanner.startswith("{|"):
                block = self.parse_annotation_block_structure()
                annotation.blocks.append(block)
                annotation.events.append(("block", block))
                continue
            break
        return annotation

    def parse_annotation_block_structure(
        self,
    ) -> list[tuple[IRI, list[tuple[Node, AnnotationData]]]]:
        """Parse annotation block structure from the current input and return the result."""
        self.scanner.expect("{|")
        skip_ws_comments(self.scanner)
        pairs = self.parse_pairs_structure(terminators=("|}",), allow_a_verb=True)
        skip_ws_comments(self.scanner)
        self.scanner.expect("|}", "expected '|}' to close annotation block")
        return pairs

    def parse_subject(self) -> IRI | BNode:
        """Parse subject from the current input and return the result."""
        if self.scanner.peek() == "(":
            return self.parse_collection_subject()
        if self.scanner.peek() == "[":
            return self.parse_blank_node_property_list()
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        return self.parse_iri()

    def parse_verb(self, allow_a: bool) -> IRI:
        """Parse verb from the current input and return the result."""
        if allow_a and is_keyword(self.scanner, "a"):
            self.scanner.expect("a")
            return IRI(RDF_TYPE_IRI)
        return self.parse_predicate()

    def parse_predicate(self) -> IRI:
        """Parse predicate from the current input and return the result."""
        return self.parse_iri()

    def parse_object(self) -> Node:
        """Parse object from the current input and return the result."""
        if self.scanner.startswith("<<("):
            return self.parse_triple_term()
        if self.scanner.startswith("<<"):
            ref_subject, _ = self.parse_reified_triple(needs_subject_reference=False)
            return ref_subject
        if self.scanner.peek() == "[":
            return self.parse_blank_node_property_list()
        if self.scanner.peek() == "(":
            return self.parse_collection_subject()
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()

        ch = self.scanner.peek()
        if ch == '"':
            return self.parse_rdf_literal(double_quote=True)
        if ch == "'":
            return self.parse_rdf_literal(double_quote=False)
        if is_keyword(self.scanner, "true"):
            self.scanner.expect("true")
            return Literal(value="true", datatype=XSD_BOOLEAN_IRI)
        if is_keyword(self.scanner, "false"):
            self.scanner.expect("false")
            return Literal(value="false", datatype=XSD_BOOLEAN_IRI)
        if ch in "+-." or ch.isdigit():
            numeric = self.try_parse_numeric_literal()
            if numeric is not None:
                return numeric
        return self.parse_iri()

    def try_parse_numeric_literal(self) -> Literal | None:
        """Try to parse numeric literal from the current input."""
        tail = self.scanner.text[self.scanner.i :]
        patterns = (
            (
                re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][+-]?[0-9]+"),
                XSD_DOUBLE_IRI,
            ),
            (re.compile(r"[+-]?(?:[0-9]*\.[0-9]+)"), XSD_DECIMAL_IRI),
            (re.compile(r"[+-]?[0-9]+"), XSD_INTEGER_IRI),
        )
        for regex, datatype in patterns:
            match = regex.match(tail)
            if not match:
                continue
            token = match.group(0)
            nxt = tail[len(token) : len(token) + 1]
            if nxt and not (is_name_boundary(nxt) or nxt in ")]};,|"):
                continue
            for _ in token:
                self.scanner.advance()
            return Literal(value=token, datatype=datatype)
        return None

    def parse_rdf_literal(self, double_quote: bool) -> Literal:
        """Parse RDF literal from the current input and return the result."""
        quote = '"' if double_quote else "'"
        if self.scanner.startswith(quote * 3):
            value = parse_long_string(
                self.scanner, quote, allow_single_quote_escape=True
            )
        else:
            value = parse_short_string(
                self.scanner, quote, allow_single_quote_escape=True
            )

        lang = None
        direction = None
        datatype = None
        if self.scanner.peek() == "@":
            lang, direction = parse_lang_dir(self.scanner)
        elif self.scanner.consume("^^"):
            datatype = self.parse_iri().value
            if datatype in (RDF_LANG_STRING_IRI, RDF_DIR_LANG_STRING_IRI):
                self.scanner.error(
                    "rdf:langString and rdf:dirLangString datatypes are not allowed"
                )
        return Literal(value=value, lang=lang, direction=direction, datatype=datatype)

    def parse_collection_subject(self) -> IRI | BNode:
        """Parse collection subject from the current input and return the result."""
        self.scanner.expect("(")
        skip_ws_comments(self.scanner)
        items: list[Node] = []
        while not self.scanner.consume(")"):
            item = self.parse_object()
            items.append(item)
            skip_ws_comments(self.scanner)
            if self.scanner.eof():
                self.scanner.error("unterminated collection")
        if not items:
            return IRI(RDF_NIL_IRI)

        head = self.new_bnode()
        current = head
        for idx, item in enumerate(items):
            self.emit(current, IRI(RDF_FIRST_IRI), item)
            if idx == len(items) - 1:
                self.emit(current, IRI(RDF_REST_IRI), IRI(RDF_NIL_IRI))
            else:
                nxt = self.new_bnode()
                self.emit(current, IRI(RDF_REST_IRI), nxt)
                current = nxt
        return head

    def parse_blank_node_property_list(self) -> BNode:
        """Parse blank-node property list from the current input and return the result."""
        self.scanner.expect("[")
        skip_ws_comments(self.scanner)
        subject = self.new_bnode()
        if self.scanner.consume("]"):
            return subject
        pairs = self.parse_pairs_structure(terminators=("]",), allow_a_verb=True)
        self.emit_pairs_from_structure(subject, pairs)
        skip_ws_comments(self.scanner)
        self.scanner.expect("]", "expected ']' to close blank node property list")
        return subject

    def parse_reified_triple(
        self, needs_subject_reference: bool
    ) -> tuple[IRI | BNode, TripleTerm]:
        """Parse reified triple syntax from the current input and return the result."""
        self.scanner.expect("<<")
        skip_ws_comments(self.scanner)
        subject = self.parse_rt_subject()
        skip_ws_comments(self.scanner)
        predicate = self.parse_verb(allow_a=True)
        skip_ws_comments(self.scanner)
        obj = self.parse_rt_object()
        skip_ws_comments(self.scanner)

        reifier: IRI | BNode | None = None
        if self.scanner.consume("~"):
            skip_ws_comments(self.scanner)
            if self.can_start_iri_or_blanknode():
                reifier = self.parse_iri_or_blanknode()
            else:
                reifier = self.new_bnode()
            skip_ws_comments(self.scanner)

        self.scanner.expect(">>", "expected '>>' to close reified triple")

        term = TripleTerm(subject=subject, predicate=predicate, object=obj)
        if reifier is not None:
            ref_subject = reifier
        else:
            ref_subject = self.new_bnode()
        self.emit(ref_subject, IRI(RDF_REIFIES_IRI), term)
        return ref_subject, term

    def parse_rt_subject(self) -> IRI | BNode | TripleTerm:
        """Parse reified-triple subject from the current input and return the result."""
        if self.scanner.startswith("<<") and not self.scanner.startswith("<<("):
            _, term = self.parse_reified_triple(needs_subject_reference=False)
            return term
        if self.scanner.peek() == "[":
            return self.parse_rt_empty_blank_node()
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        return self.parse_iri()

    def parse_rt_object(self) -> Node:
        """Parse reified-triple object from the current input and return the result."""
        if self.scanner.startswith("<<("):
            return self.parse_triple_term()
        if self.scanner.startswith("<<"):
            ref_subject, _ = self.parse_reified_triple(needs_subject_reference=False)
            return ref_subject
        if self.scanner.peek() == "[":
            return self.parse_rt_empty_blank_node()
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        ch = self.scanner.peek()
        if ch in ('"', "'"):
            return self.parse_rdf_literal(double_quote=(ch == '"'))
        if is_keyword(self.scanner, "true"):
            self.scanner.expect("true")
            return Literal(value="true", datatype=XSD_BOOLEAN_IRI)
        if is_keyword(self.scanner, "false"):
            self.scanner.expect("false")
            return Literal(value="false", datatype=XSD_BOOLEAN_IRI)
        if ch in "+-." or ch.isdigit():
            numeric = self.try_parse_numeric_literal()
            if numeric is not None:
                return numeric
        return self.parse_iri()

    def parse_triple_term(self) -> TripleTerm:
        """Parse triple-term node from the current input and return the result."""
        self.scanner.expect("<<(")
        skip_ws_comments(self.scanner)
        subject = self.parse_tt_subject()
        skip_ws_comments(self.scanner)
        predicate = self.parse_verb(allow_a=True)
        skip_ws_comments(self.scanner)
        obj = self.parse_tt_object()
        skip_ws_comments(self.scanner)
        self.scanner.expect(")>>", "expected ')>>' to close triple term")
        return TripleTerm(subject=subject, predicate=predicate, object=obj)

    def parse_tt_subject(self) -> IRI | BNode:
        """Parse triple-term subject from the current input and return the result."""
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        return self.parse_iri()

    def parse_tt_object(self) -> Node:
        """Parse triple-term object from the current input and return the result."""
        if self.scanner.startswith("<<("):
            return self.parse_triple_term()
        if self.scanner.startswith("<<"):
            ref_subject, _ = self.parse_reified_triple(needs_subject_reference=False)
            return ref_subject
        if self.scanner.peek() == "[":
            return self.parse_blank_node_property_list()
        if self.scanner.peek() == "(":
            return self.parse_collection_subject()
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        ch = self.scanner.peek()
        if ch in ('"', "'"):
            return self.parse_rdf_literal(double_quote=(ch == '"'))
        if is_keyword(self.scanner, "true"):
            self.scanner.expect("true")
            return Literal(value="true", datatype=XSD_BOOLEAN_IRI)
        if is_keyword(self.scanner, "false"):
            self.scanner.expect("false")
            return Literal(value="false", datatype=XSD_BOOLEAN_IRI)
        if ch in "+-." or ch.isdigit():
            numeric = self.try_parse_numeric_literal()
            if numeric is not None:
                return numeric
        return self.parse_iri()

    def parse_iri_or_blanknode(self) -> IRI | BNode:
        """Parse IRI or blanknode from the current input and return the result."""
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        return self.parse_iri()

    def can_start_iri_or_blanknode(self) -> bool:
        """Return whether the next token can start an IRI or blank node."""
        if self.scanner.startswith("_:"):
            return True
        ch = self.scanner.peek()
        return ch == "<" or ch == ":" or is_pn_chars_base(ch)

    def parse_rt_empty_blank_node(self) -> BNode:
        """Parse reified-triple empty blank node from the current input and return the result."""
        self.scanner.expect("[")
        skip_ws_comments(self.scanner)
        if not self.scanner.consume("]"):
            self.scanner.error("reified triple allows only empty blank node []")
        return self.new_bnode()

    def parse_blank_node_label(self) -> BNode:
        """Parse blank node label from the current input and return the result."""
        self.scanner.expect("_:")
        if not (is_pn_chars_u(self.scanner.peek()) or self.scanner.peek().isdigit()):
            self.scanner.error("invalid blank node label")
        chars = [self.scanner.advance()]
        while True:
            ch = self.scanner.peek()
            if not ch:
                break
            if is_pn_chars(ch):
                chars.append(self.scanner.advance())
                continue
            if ch == ".":
                nxt = self.scanner.peek(1)
                if nxt and (nxt == "." or is_pn_chars(nxt)):
                    chars.append(self.scanner.advance())
                    continue
                break
            break
        label = "".join(chars)
        self._reserved_labels.add(label)
        return BNode(label)

    def parse_iri(self) -> IRI:
        """Parse IRI from the current input and return the result."""
        if self.scanner.peek() == "<":
            return IRI(self.parse_iri_ref_turtle())
        return IRI(self.parse_prefixed_name())

    def parse_iri_ref_turtle(self) -> str:
        """Parse Turtle IRI reference from the current input and return the result."""
        try:
            iri = parse_iri_ref(self.scanner, require_absolute=False, allow_empty=True)
        except ValueError as exc:
            self.scanner.error(str(exc))
            raise AssertionError("unreachable")
        if self.base_iri is not None and not urlsplit(iri).scheme:
            return resolve_iri_reference(self.base_iri, iri)
        return iri

    def parse_prefixed_name(self) -> str:
        """Parse prefixed name from the current input and return the result."""
        prefix = self.parse_pname_ns()
        local = ""
        if not is_name_boundary(self.scanner.peek()):
            local = self.parse_pn_local()
        if prefix not in self.prefixes:
            self.scanner.error(f"undeclared prefix '{prefix}'")
        return self.prefixes[prefix] + local

    def parse_pname_ns(self) -> str:
        """Parse PName namespace prefix from the current input and return the result."""
        if self.scanner.consume(":"):
            return ""
        if not is_pn_chars_base(self.scanner.peek()):
            self.scanner.error("invalid prefix name")
        chars = [self.scanner.advance()]
        while True:
            ch = self.scanner.peek()
            if not ch:
                self.scanner.error("unterminated prefix name")
            if ch == ":":
                self.scanner.advance()
                if chars[-1] == ".":
                    self.scanner.error("prefix cannot end with '.'")
                return "".join(chars)
            if is_pn_chars(ch) or ch == ".":
                chars.append(self.scanner.advance())
                continue
            self.scanner.error("invalid character in prefix")

    def parse_pn_local(self) -> str:
        """Parse PN_LOCAL text from the current input and return the result."""
        parts: list[str] = []
        endable = False

        first = self.scanner.peek()
        if first == "%":
            parts.append(self.parse_percent())
            endable = True
        elif first == "\\":
            parts.append(self.parse_pn_local_escape())
            endable = True
        elif first == ":" or first.isdigit() or is_pn_chars_u(first):
            parts.append(self.scanner.advance())
            endable = True
        else:
            self.scanner.error("invalid local name")

        while True:
            ch = self.scanner.peek()
            if not ch:
                break
            if ch == "%":
                parts.append(self.parse_percent())
                endable = True
                continue
            if ch == "\\":
                parts.append(self.parse_pn_local_escape())
                endable = True
                continue
            if ch == ":" or is_pn_chars(ch):
                parts.append(self.scanner.advance())
                endable = True
                continue
            if ch == ".":
                nxt = self.scanner.peek(1)
                if nxt and (
                    nxt == "."
                    or nxt == "%"
                    or nxt == "\\"
                    or nxt == ":"
                    or is_pn_chars(nxt)
                ):
                    parts.append(self.scanner.advance())
                    endable = False
                    continue
                break
            break

        if not endable:
            self.scanner.error("local name cannot end with '.'")
        return "".join(parts)

    def parse_percent(self) -> str:
        """Parse percent from the current input and return the result."""
        self.scanner.expect("%")
        a = self.scanner.peek()
        if not is_hex(a):
            self.scanner.error("invalid percent escape")
        self.scanner.advance()
        b = self.scanner.peek()
        if not is_hex(b):
            self.scanner.error("invalid percent escape")
        self.scanner.advance()
        return f"%{a}{b}"

    def parse_pn_local_escape(self) -> str:
        """Parse PN_LOCAL escape sequence from the current input and return the result."""
        self.scanner.expect("\\")
        ch = self.scanner.peek()
        allowed = "_~.-!$&'()*+,;=/?#@%"
        if ch not in allowed:
            self.scanner.error("invalid PN_LOCAL escape")
        self.scanner.advance()
        return ch

    def can_start_verb(self) -> bool:
        """Return whether the next token can start a Turtle verb."""
        if is_keyword(self.scanner, "a"):
            return True
        ch = self.scanner.peek()
        if ch == "<" or ch == ":":
            return True
        return is_pn_chars_base(ch)


class TriGParser(TurtleParser):
    """Parser for the repository's RDF 1.2 TriG grammar."""

    def __init__(self, text: str, source: str, base_iri: str | None):
        """Initialize the `TriGParser` instance."""
        super().__init__(text=text, source=source, base_iri=base_iri)
        self.quads: list[Quad] = []
        self._current_graph: GraphLabel | None = None

    def emit(self, subject: IRI | BNode, predicate: IRI, obj: Node) -> None:
        """Append one quad to the parser output buffer."""
        self.quads.append((subject, predicate, obj, self._current_graph))

    def parse(self) -> list[Quad]:
        """Parse the current document and return parsed quads."""
        while True:
            skip_ws_comments(self.scanner)
            if self.scanner.eof():
                return self.quads
            if self.parse_directive_if_present():
                continue
            self.parse_trig_block()

    def parse_trig_block(self) -> None:
        """Parse one top-level TriG block or triples statement."""
        skip_ws_comments(self.scanner)
        if is_keyword_ci(self.scanner, "GRAPH"):
            for _ in "GRAPH":
                self.scanner.advance()
            skip_ws_comments(self.scanner)
            graph_label = self.parse_graph_label_trig()
            skip_ws_comments(self.scanner)
            self.parse_wrapped_graph(graph_label)
            return

        if self.scanner.peek() == "{":
            self.parse_wrapped_graph(None)
            return

        if self.scanner.peek() == "(" or (
            self.scanner.startswith("<<") and not self.scanner.startswith("<<(")
        ):
            self.parse_triples_statement()
            skip_ws_comments(self.scanner)
            self.scanner.expect(".", "expected '.' to end TriG triple statement")
            return

        if self.scanner.peek() == "[":
            emitted_before = len(self.quads)
            subject = self.parse_blank_node_property_list()
            emitted_in_subject = len(self.quads) != emitted_before
            skip_ws_comments(self.scanner)
            if self.scanner.peek() == "{":
                if emitted_in_subject:
                    self.scanner.error(
                        "TriG graph labels cannot be blankNodePropertyList terms"
                    )
                self.parse_wrapped_graph(subject)
                return
            if self.can_start_verb():
                pairs = self.parse_pairs_structure(
                    terminators=(".",), allow_a_verb=True
                )
                self.emit_pairs_from_structure(subject, pairs)
            elif not self.scanner.startswith("."):
                self.scanner.error("expected '{' or predicate after TriG subject")
            skip_ws_comments(self.scanner)
            self.scanner.expect(".", "expected '.' to end TriG triple statement")
            return

        subject = self.parse_trig_label_or_subject()
        skip_ws_comments(self.scanner)
        if self.scanner.peek() == "{":
            self.parse_wrapped_graph(subject)
            return

        pairs = self.parse_pairs_structure(terminators=(".",), allow_a_verb=True)
        self.emit_pairs_from_structure(subject, pairs)
        skip_ws_comments(self.scanner)
        self.scanner.expect(".", "expected '.' to end TriG triple statement")

    def parse_wrapped_graph(self, graph_label: GraphLabel | None) -> None:
        """Parse a TriG wrapped graph block in the given graph context."""
        self.scanner.expect("{")
        previous_graph = self._current_graph
        self._current_graph = graph_label
        try:
            skip_ws_comments(self.scanner)
            if self.scanner.consume("}"):
                return

            while True:
                self.parse_triples_statement_in_graph_block()
                skip_ws_comments(self.scanner)
                if self.scanner.consume("."):
                    skip_ws_comments(self.scanner)
                    if self.scanner.consume("}"):
                        return
                    continue
                self.scanner.expect(
                    "}", "expected '}' or '.' after TriG triples in graph block"
                )
                return
        finally:
            self._current_graph = previous_graph

    def parse_triples_statement_in_graph_block(self) -> None:
        """Parse a TriG triples statement inside `{...}` with `}` as a terminator."""
        skip_ws_comments(self.scanner)
        if self.scanner.startswith("<<") and not self.scanner.startswith("<<("):
            subject, _ = self.parse_reified_triple(needs_subject_reference=True)
            skip_ws_comments(self.scanner)
            if self.can_start_verb():
                pairs = self.parse_pairs_structure(
                    terminators=(".", "}"), allow_a_verb=True
                )
                self.emit_pairs_from_structure(subject, pairs)
            return

        if self.scanner.peek() == "[":
            subject = self.parse_blank_node_property_list()
            skip_ws_comments(self.scanner)
            if self.can_start_verb():
                pairs = self.parse_pairs_structure(
                    terminators=(".", "}"), allow_a_verb=True
                )
                self.emit_pairs_from_structure(subject, pairs)
            return

        subject = self.parse_subject()
        skip_ws_comments(self.scanner)
        pairs = self.parse_pairs_structure(terminators=(".", "}"), allow_a_verb=True)
        self.emit_pairs_from_structure(subject, pairs)

    def parse_trig_label_or_subject(self) -> IRI | BNode:
        """Parse a TriG labelOrSubject in non-bracket form."""
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        return self.parse_iri()

    def parse_graph_label_trig(self) -> GraphLabel:
        """Parse a TriG graph label (`iri`, `_:` label, or `[]`)."""
        if self.scanner.peek() == "[":
            self.scanner.expect("[")
            skip_ws_comments(self.scanner)
            if not self.scanner.consume("]"):
                self.scanner.error("TriG graph label blank node must be [] or _:label")
            return self.new_bnode()
        if self.scanner.startswith("_:"):
            return self.parse_blank_node_label()
        return self.parse_iri()


def parse_ntriples(text: str, source: str = "<string>") -> list[Triple]:
    """Parse N-Triples text and return a list of RDF triples."""
    parser = NTriplesParser(text=text, source=source)
    return parser.parse()


def parse_nquads(text: str, source: str = "<string>") -> list[Quad]:
    """Parse N-Quads text and return a list of RDF quads."""
    parser = NQuadsParser(text=text, source=source)
    return parser.parse()


def parse_turtle(
    text: str, source: str = "<string>", base_iri: str | None = None
) -> list[Triple]:
    """Parse Turtle text and return a list of RDF triples."""
    parser = TurtleParser(text=text, source=source, base_iri=base_iri)
    return parser.parse()


def parse_trig(
    text: str, source: str = "<string>", base_iri: str | None = None
) -> list[Quad]:
    """Parse TriG text and return a list of RDF quads."""
    parser = TriGParser(text=text, source=source, base_iri=base_iri)
    return parser.parse()


def parse_cli_graph_label(value: str, *, option_name: str) -> GraphLabel:
    """Parse CLI graph-label option as an absolute IRI or blank node label."""
    raw = value.strip()
    if not raw:
        raise ValueError(f"{option_name} requires a non-empty value")

    if raw.startswith("_:"):
        parser = NTriplesParser(text=raw, source=option_name)
        label = parser.parse_blank_node_label()
        if not parser.scanner.eof():
            parser.scanner.error("unexpected trailing characters in graph label")
        return label

    try:
        validate_iri(raw, require_absolute=True, allow_empty=False)
    except ValueError as exc:
        raise ValueError(
            f"invalid {option_name} value '{value}': expected absolute IRI or _:label ({exc})"
        ) from exc
    return IRI(raw)


def triples_to_quads(
    triples: list[Triple],
    *,
    graph_label: GraphLabel | None = None,
) -> list[Quad]:
    """Lift triples into one dataset graph for internal dataset processing."""
    return [
        (subject, predicate, obj, graph_label) for subject, predicate, obj in triples
    ]


def quads_to_default_graph_triples(
    quads: list[Quad],
    *,
    target_format: str,
) -> list[Triple]:
    """Drop graph labels after verifying that only the default graph is present."""
    triples: list[Triple] = []
    for subject, predicate, obj, graph_label in quads:
        if graph_label is not None:
            raise ValueError(
                f"cannot serialize named graphs to {target_format}; dataset contains non-default graph statements"
            )
        triples.append((subject, predicate, obj))
    return triples


def quads_to_graph_triples(
    quads: list[Quad],
    *,
    target_format: str,
    policy: str = "strict",
    selected_graph: GraphLabel | None = None,
) -> list[Triple]:
    """Project a dataset into a graph according to an explicit lossy policy."""
    if policy == "strict":
        return quads_to_default_graph_triples(quads, target_format=target_format)

    if policy == "default-only":
        return [
            (subject, predicate, obj)
            for subject, predicate, obj, graph_label in quads
            if graph_label is None
        ]

    if policy == "select":
        if selected_graph is None:
            raise ValueError(
                "--dataset-to-graph-policy select requires --graph <IRI|_:label>"
            )
        return [
            (subject, predicate, obj)
            for subject, predicate, obj, graph_label in quads
            if graph_label == selected_graph
        ]

    if policy == "union":
        return [(subject, predicate, obj) for subject, predicate, obj, _ in quads]

    raise ValueError(f"unsupported dataset-to-graph policy: {policy}")


def format_node_nt(node: Node) -> str:
    """Format an RDF node using N-Triples syntax."""
    if isinstance(node, IRI):
        return encode_iri_ref(node.value)
    if isinstance(node, BNode):
        return f"_:{node.label}"
    if isinstance(node, Literal):
        base = f'"{escape_string_value(node.value)}"'
        if node.lang is not None:
            if node.direction is not None:
                return f"{base}@{node.lang}--{node.direction}"
            return f"{base}@{node.lang}"
        if node.datatype is not None:
            if node.datatype == f"{XSD_NS}string":
                return base
            return f"{base}^^{encode_iri_ref(node.datatype)}"
        return base
    if isinstance(node, TripleTerm):
        subject = format_node_nt(node.subject)
        predicate = format_node_nt(node.predicate)
        obj = format_node_nt(node.object)
        return f"<<( {subject} {predicate} {obj} )>>"
    raise TypeError(f"unsupported node type: {type(node)!r}")


def serialize_ntriples(triples: list[Triple]) -> str:
    """Serialize triples to deterministic N-Triples text."""
    lines: list[str] = []
    for subject, predicate, obj in triples:
        s = format_node_nt(subject)
        p = format_node_nt(predicate)
        o = format_node_nt(obj)
        lines.append(f"{s} {p} {o} .")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def format_graph_label_nt(graph_label: GraphLabel) -> str:
    """Format an RDF graph label using N-Quads syntax."""
    if isinstance(graph_label, IRI):
        return encode_iri_ref(graph_label.value)
    if isinstance(graph_label, BNode):
        return f"_:{graph_label.label}"
    raise TypeError(f"invalid graph label type for N-Quads: {type(graph_label)!r}")


def serialize_nquads(quads: list[Quad]) -> str:
    """Serialize quads to deterministic N-Quads text."""
    lines: list[str] = []
    for subject, predicate, obj, graph_label in quads:
        s = format_node_nt(subject)
        p = format_node_nt(predicate)
        o = format_node_nt(obj)
        if graph_label is None:
            lines.append(f"{s} {p} {o} .")
        else:
            g = format_graph_label_nt(graph_label)
            lines.append(f"{s} {p} {o} {g} .")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


KNOWN_PREFIXES = {
    RDF_NS: "rdf",
    XSD_NS: "xsd",
}


@dataclass(frozen=True)
class TurtleSerializationOptions:
    """Options controlling Turtle serialization behavior and readability."""
    style: str = "readable"
    lists: str = "auto"
    auto_prefixes: str = "on"
    manual_prefixes: tuple[tuple[str, str], ...] = ()
    output_base: str | None = None


def split_iri_for_prefix(iri: str) -> tuple[str, str] | None:
    """Split an IRI into a candidate namespace/local-name pair."""
    if not iri:
        return None
    scheme_sep = iri.find(":")
    if scheme_sep < 0:
        return None

    candidates: list[int] = []
    for idx, ch in enumerate(iri):
        if ch in "#/":
            candidates.append(idx)
        elif ch == ":" and idx > scheme_sep:
            candidates.append(idx)

    if not candidates:
        return None

    cut = max(candidates)
    return iri[: cut + 1], iri[cut + 1 :]


def escape_pn_local(local: str) -> str | None:
    """Escape PN_LOCAL text."""
    if local == "":
        return ""
    out: list[str] = []
    endable = False
    i = 0
    while i < len(local):
        ch = local[i]
        first = i == 0
        last = i == len(local) - 1

        if (
            ch == "%"
            and i + 2 < len(local)
            and is_hex(local[i + 1])
            and is_hex(local[i + 2])
        ):
            out.append(local[i : i + 3])
            endable = True
            i += 3
            continue

        if ch == ".":
            if last:
                out.append("\\.")
                endable = True
            else:
                out.append(".")
                endable = False
            i += 1
            continue

        if first and (ch == ":" or ch.isdigit() or is_pn_chars_u(ch)):
            out.append(ch)
            endable = True
            i += 1
            continue
        if not first and (ch == ":" or is_pn_chars(ch)):
            out.append(ch)
            endable = True
            i += 1
            continue

        if ch in "_~.-!$&'()*+,;=/?#@%":
            out.append("\\" + ch)
            endable = True
            i += 1
            continue

        return None

    if not endable:
        return None
    return "".join(out)


def is_valid_prefix_label(prefix: str) -> bool:
    """Return whether a string is a valid Turtle prefix label."""
    if prefix == "":
        return True
    if ":" in prefix:
        return False
    if not is_pn_chars_base(prefix[0]):
        return False
    if len(prefix) == 1:
        return True
    if prefix[-1] == ".":
        return False
    for ch in prefix[1:]:
        if ch == "." or is_pn_chars(ch):
            continue
        return False
    return True


def parse_prefix_binding(raw: str) -> tuple[str, str]:
    """Parse `PREFIX=IRI` binding from the current input and return the result."""
    if "=" not in raw:
        raise ValueError(f"invalid --prefix value '{raw}', expected PREFIX=IRI")
    prefix, iri = raw.split("=", 1)
    if not is_valid_prefix_label(prefix):
        raise ValueError(f"invalid prefix label '{prefix}' in --prefix")
    try:
        validate_iri(iri, require_absolute=False, allow_empty=True)
    except ValueError as exc:
        raise ValueError(f"invalid IRI for --prefix '{raw}': {exc}") from exc
    return prefix, iri


def normalize_manual_prefixes(raw_values: list[str]) -> tuple[tuple[str, str], ...]:
    """Normalize manual prefix bindings."""
    ordered: list[tuple[str, str]] = []
    by_prefix: dict[str, str] = {}
    for raw in raw_values:
        prefix, iri = parse_prefix_binding(raw)
        existing = by_prefix.get(prefix)
        if existing is not None and existing != iri:
            raise ValueError(
                f"conflicting --prefix for '{prefix}': '{existing}' vs '{iri}'"
            )
        if existing is None:
            by_prefix[prefix] = iri
            ordered.append((prefix, iri))
    return tuple(ordered)


def collect_list_compaction(
    triples: list[Triple],
) -> tuple[dict[BNode, list[Node]], set[int]]:
    """Collect RDF list compaction candidates."""
    by_subject: dict[BNode, dict[str, list[tuple[Node, int]]]] = {}
    incoming_total: dict[BNode, int] = {}
    incoming_rest: dict[BNode, int] = {}

    for idx, (subject, predicate, obj) in enumerate(triples):
        if isinstance(subject, BNode):
            pred_map = by_subject.setdefault(subject, {})
            pred_map.setdefault(predicate.value, []).append((obj, idx))
        if isinstance(obj, BNode):
            incoming_total[obj] = incoming_total.get(obj, 0) + 1
            if predicate.value == RDF_REST_IRI:
                incoming_rest[obj] = incoming_rest.get(obj, 0) + 1

    compacted_heads: dict[BNode, list[Node]] = {}
    removed_indices: set[int] = set()

    for head, pred_map in by_subject.items():
        non_rest_refs = incoming_total.get(head, 0) - incoming_rest.get(head, 0)
        if non_rest_refs != 1:
            continue

        items: list[Node] = []
        chain_indices: list[int] = []
        visited: set[BNode] = set()
        current: BNode | IRI = head
        valid = True

        while True:
            if not isinstance(current, BNode):
                valid = False
                break
            if current in visited:
                valid = False
                break
            visited.add(current)

            current_pred_map = by_subject.get(current)
            if current_pred_map is None:
                valid = False
                break
            if set(current_pred_map) != {RDF_FIRST_IRI, RDF_REST_IRI}:
                valid = False
                break

            first_entries = current_pred_map[RDF_FIRST_IRI]
            rest_entries = current_pred_map[RDF_REST_IRI]
            if len(first_entries) != 1 or len(rest_entries) != 1:
                valid = False
                break

            first_obj, first_idx = first_entries[0]
            rest_obj, rest_idx = rest_entries[0]
            items.append(first_obj)
            chain_indices.extend([first_idx, rest_idx])

            if current != head:
                if (
                    incoming_total.get(current, 0) != 1
                    or incoming_rest.get(current, 0) != 1
                ):
                    valid = False
                    break

            if isinstance(rest_obj, IRI) and rest_obj.value == RDF_NIL_IRI:
                break
            if not isinstance(rest_obj, BNode):
                valid = False
                break
            current = rest_obj

        if not valid:
            continue
        if any(idx in removed_indices for idx in chain_indices):
            continue

        compacted_heads[head] = items
        removed_indices.update(chain_indices)

    return compacted_heads, removed_indices


def iter_iris_in_rendered_node(
    node: Node,
    list_heads: dict[BNode, list[Node]],
    active_lists: set[BNode] | None = None,
) -> Iterable[str]:
    """Yield iris in rendered node."""
    if active_lists is None:
        active_lists = set()

    if isinstance(node, IRI):
        yield node.value
        return
    if isinstance(node, BNode):
        if node in list_heads and node not in active_lists:
            active_lists.add(node)
            for item in list_heads[node]:
                yield from iter_iris_in_rendered_node(item, list_heads, active_lists)
            active_lists.remove(node)
        return
    if isinstance(node, Literal):
        if node.datatype is not None:
            yield node.datatype
        return
    if isinstance(node, TripleTerm):
        if isinstance(node.subject, IRI):
            yield node.subject.value
        elif isinstance(node.subject, BNode) and node.subject in list_heads:
            yield from iter_iris_in_rendered_node(
                node.subject, list_heads, active_lists
            )
        if node.predicate.value != RDF_TYPE_IRI:
            yield node.predicate.value
        yield from iter_iris_in_rendered_node(node.object, list_heads, active_lists)
        return
    raise TypeError(f"unsupported node type: {type(node)!r}")


def choose_turtle_prefixes(
    triples: list[Triple],
    list_heads: dict[BNode, list[Node]],
) -> dict[str, str]:
    """Choose automatic Turtle prefix assignments."""
    stats: dict[str, dict[str, int]] = {}

    def register_iri(iri: str, is_predicate: bool) -> None:
        """Record namespace usage statistics for one IRI candidate."""
        split = split_iri_for_prefix(iri)
        if split is None:
            return
        namespace, local = split
        escaped = escape_pn_local(local)
        if escaped is None:
            return
        stat = stats.setdefault(namespace, {"count": 0, "pred_count": 0})
        stat["count"] += 1
        if is_predicate:
            stat["pred_count"] += 1

    for subject, predicate, obj in triples:
        for iri in iter_iris_in_rendered_node(subject, list_heads):
            register_iri(iri, is_predicate=False)
        if predicate.value != RDF_TYPE_IRI:
            register_iri(predicate.value, is_predicate=True)
        for iri in iter_iris_in_rendered_node(obj, list_heads):
            register_iri(iri, is_predicate=False)

    selected: list[str] = []
    for namespace, stat in stats.items():
        if namespace in KNOWN_PREFIXES:
            selected.append(namespace)
            continue
        if stat["pred_count"] > 0 or stat["count"] >= 2:
            selected.append(namespace)

    assigned: dict[str, str] = {}
    used_prefixes: set[str] = set()

    for namespace in selected:
        known = KNOWN_PREFIXES.get(namespace)
        if known is None or known in used_prefixes:
            continue
        assigned[namespace] = known
        used_prefixes.add(known)

    dynamic_namespaces = [ns for ns in selected if ns not in assigned]
    dynamic_namespaces.sort(
        key=lambda ns: (-stats[ns]["pred_count"], -stats[ns]["count"], ns)
    )

    serial = 1
    for namespace in dynamic_namespaces:
        while True:
            candidate = f"ns{serial}"
            serial += 1
            if candidate not in used_prefixes:
                break
        assigned[namespace] = candidate
        used_prefixes.add(candidate)

    ordered = sorted(assigned.items(), key=lambda item: item[1])
    return {prefix: namespace for namespace, prefix in ordered}


def merge_prefix_maps(
    manual_prefixes: tuple[tuple[str, str], ...],
    auto_prefixes: dict[str, str],
    auto_enabled: bool,
) -> dict[str, str]:
    """Merge prefix maps."""
    merged: dict[str, str] = {}
    used_namespaces: set[str] = set()

    for prefix, namespace in manual_prefixes:
        merged[prefix] = namespace
        used_namespaces.add(namespace)

    if not auto_enabled:
        return merged

    for prefix, namespace in auto_prefixes.items():
        if prefix in merged:
            continue
        if namespace in used_namespaces:
            continue
        merged[prefix] = namespace
        used_namespaces.add(namespace)
    return merged


def format_iri_turtle(iri: str, prefixes: dict[str, str]) -> str:
    """Format IRI turtle for output."""
    split = split_iri_for_prefix(iri)
    if split is not None:
        namespace, local = split
        escaped = escape_pn_local(local)
        for prefix, base in prefixes.items():
            if base == namespace and escaped is not None:
                if escaped:
                    return f"{prefix}:{escaped}"
                return f"{prefix}:"
    return encode_iri_ref(iri)


def format_predicate_turtle(predicate: IRI, prefixes: dict[str, str]) -> str:
    """Format predicate turtle for output."""
    if predicate.value == RDF_TYPE_IRI:
        return "a"
    return format_iri_turtle(predicate.value, prefixes)


def format_node_turtle(
    node: Node,
    prefixes: dict[str, str],
    list_heads: dict[BNode, list[Node]],
    active_lists: set[BNode] | None = None,
) -> str:
    """Format node turtle for output."""
    if active_lists is None:
        active_lists = set()

    if isinstance(node, IRI):
        return format_iri_turtle(node.value, prefixes)
    if isinstance(node, BNode):
        if node in list_heads and node not in active_lists:
            active_lists.add(node)
            inner = " ".join(
                format_node_turtle(item, prefixes, list_heads, active_lists)
                for item in list_heads[node]
            )
            active_lists.remove(node)
            return f"({inner})"
        return f"_:{node.label}"
    if isinstance(node, Literal):
        base = f'"{escape_string_value(node.value)}"'
        if node.lang is not None:
            if node.direction is not None:
                return f"{base}@{node.lang}--{node.direction}"
            return f"{base}@{node.lang}"
        if node.datatype is not None:
            datatype = format_iri_turtle(node.datatype, prefixes)
            return f"{base}^^{datatype}"
        return base
    if isinstance(node, TripleTerm):
        subject = format_node_turtle(node.subject, prefixes, list_heads, active_lists)
        predicate = format_predicate_turtle(node.predicate, prefixes)
        obj = format_node_turtle(node.object, prefixes, list_heads, active_lists)
        return f"<<({subject} {predicate} {obj})>>"
    raise TypeError(f"unsupported node type: {type(node)!r}")


def format_graph_label_turtle(graph_label: GraphLabel, prefixes: dict[str, str]) -> str:
    """Format a graph label for TriG output."""
    if isinstance(graph_label, IRI):
        return format_iri_turtle(graph_label.value, prefixes)
    if isinstance(graph_label, BNode):
        return f"_:{graph_label.label}"
    raise TypeError(f"invalid graph label type for TriG: {type(graph_label)!r}")


def build_turtle_statement_blocks(
    triples: list[Triple],
    prefixes: dict[str, str],
    list_heads: dict[BNode, list[Node]],
) -> list[str]:
    """Render grouped Turtle subject blocks without directives/prefix headers."""
    grouped: dict[IRI | BNode, dict[IRI, list[Node]]] = {}
    for subject, predicate, obj in triples:
        predicates = grouped.setdefault(subject, {})
        objects = predicates.setdefault(predicate, [])
        objects.append(obj)

    blocks: list[str] = []
    for subject, predicates in grouped.items():
        subject_text = format_node_turtle(subject, prefixes, list_heads)
        predicate_parts: list[str] = []
        for predicate, objects in predicates.items():
            predicate_text = format_predicate_turtle(predicate, prefixes)
            object_text = ", ".join(
                format_node_turtle(obj, prefixes, list_heads) for obj in objects
            )
            predicate_parts.append(f"{predicate_text} {object_text}")

        block = f"{subject_text} {predicate_parts[0]}"
        for part in predicate_parts[1:]:
            block += f"\n    ; {part}"
        block += " ."
        blocks.append(block)
    return blocks


def indent_multiline_block(text: str, prefix: str) -> str:
    """Indent every line of a multi-line text block."""
    return "\n".join(prefix + line for line in text.splitlines())


def serialize_turtle_with_meta(
    triples: list[Triple],
    options: TurtleSerializationOptions | None = None,
) -> tuple[str, dict[str, int]]:
    """Serialize triples to Turtle and return serialization metadata counters."""
    opts = options or TurtleSerializationOptions()
    if opts.style not in {"readable", "minimal"}:
        raise ValueError(f"unsupported turtle style: {opts.style}")

    lines: list[str] = []
    meta = {
        "prefixes_emitted": 0,
        "lists_compacted": 0,
        "triples_serialized": len(triples),
    }

    if opts.output_base is not None:
        lines.append(f"@base {encode_iri_ref(opts.output_base)} .")

    if opts.style == "minimal":
        for subject, predicate, obj in triples:
            s = format_node_nt(subject)
            p = format_node_nt(predicate)
            o = format_node_nt(obj)
            lines.append(f"{s} {p} {o} .")
        if not lines:
            return "", meta
        return "\n".join(lines) + "\n", meta

    if opts.lists == "auto":
        list_heads, removed_indices = collect_list_compaction(triples)
    elif opts.lists == "off":
        list_heads, removed_indices = {}, set()
    else:
        raise ValueError(f"unsupported lists mode: {opts.lists}")
    meta["lists_compacted"] = len(list_heads)

    visible_triples = [
        triple for idx, triple in enumerate(triples) if idx not in removed_indices
    ]
    meta["triples_serialized"] = len(visible_triples)

    auto_prefixes = choose_turtle_prefixes(visible_triples, list_heads)
    prefixes = merge_prefix_maps(
        manual_prefixes=opts.manual_prefixes,
        auto_prefixes=auto_prefixes,
        auto_enabled=(opts.auto_prefixes == "on"),
    )
    meta["prefixes_emitted"] = len(prefixes)

    if prefixes:
        if lines:
            lines.append("")
        for prefix, namespace in prefixes.items():
            lines.append(f"@prefix {prefix}: {encode_iri_ref(namespace)} .")
        lines.append("")
    elif opts.output_base is not None:
        lines.append("")

    lines.extend(build_turtle_statement_blocks(visible_triples, prefixes, list_heads))

    if not lines:
        return "", meta
    return "\n".join(lines) + "\n", meta


def serialize_turtle(
    triples: list[Triple],
    options: TurtleSerializationOptions | None = None,
) -> str:
    """Serialize triples to Turtle text using the selected options."""
    text, _ = serialize_turtle_with_meta(triples, options=options)
    return text


def serialize_trig_with_meta(
    quads: list[Quad],
    options: TurtleSerializationOptions | None = None,
) -> tuple[str, dict[str, int]]:
    """Serialize quads to TriG and return serialization metadata counters."""
    opts = options or TurtleSerializationOptions()
    if opts.style not in {"readable", "minimal"}:
        raise ValueError(f"unsupported turtle style: {opts.style}")

    grouped: dict[GraphLabel | None, list[Triple]] = {}
    for subject, predicate, obj, graph_label in quads:
        triples = grouped.setdefault(graph_label, [])
        triples.append((subject, predicate, obj))

    lines: list[str] = []
    meta = {
        "prefixes_emitted": 0,
        "lists_compacted": 0,
        "triples_serialized": len(quads),
        "quads_serialized": len(quads),
        "graphs_serialized": len(grouped),
    }

    if opts.output_base is not None:
        lines.append(f"@base {encode_iri_ref(opts.output_base)} .")

    if opts.style == "minimal":
        for graph_label, triples in grouped.items():
            if graph_label is None:
                for subject, predicate, obj in triples:
                    s = format_node_nt(subject)
                    p = format_node_nt(predicate)
                    o = format_node_nt(obj)
                    lines.append(f"{s} {p} {o} .")
                continue

            graph_text = format_graph_label_nt(graph_label)
            body_lines = []
            for subject, predicate, obj in triples:
                s = format_node_nt(subject)
                p = format_node_nt(predicate)
                o = format_node_nt(obj)
                body_lines.append(f"{s} {p} {o} .")
            if not body_lines:
                continue
            body = indent_multiline_block("\n".join(body_lines), "    ")
            lines.append(f"{graph_text} {{\n{body}\n}}")

        if not lines:
            return "", meta
        return "\n".join(lines) + "\n", meta

    if opts.lists == "auto":
        plans: dict[GraphLabel | None, tuple[list[Triple], dict[BNode, list[Node]]]] = {}
        combined_visible: list[Triple] = []
        combined_list_heads: dict[BNode, list[Node]] = {}
        total_visible = 0
        for graph_label, triples in grouped.items():
            list_heads, removed_indices = collect_list_compaction(triples)
            visible_triples = [
                triple for idx, triple in enumerate(triples) if idx not in removed_indices
            ]
            plans[graph_label] = (visible_triples, list_heads)
            combined_visible.extend(visible_triples)
            combined_list_heads.update(list_heads)
            total_visible += len(visible_triples)
            meta["lists_compacted"] += len(list_heads)
        meta["triples_serialized"] = total_visible
    elif opts.lists == "off":
        plans = {
            graph_label: (triples, {})
            for graph_label, triples in grouped.items()
        }
    else:
        raise ValueError(f"unsupported lists mode: {opts.lists}")

    if opts.lists == "off":
        combined_visible = []
        combined_list_heads = {}
        total_visible = 0
        for visible_triples, list_heads in plans.values():
            combined_visible.extend(visible_triples)
            combined_list_heads.update(list_heads)
            total_visible += len(visible_triples)
        meta["triples_serialized"] = total_visible

    auto_prefixes = choose_turtle_prefixes(combined_visible, combined_list_heads)
    prefixes = merge_prefix_maps(
        manual_prefixes=opts.manual_prefixes,
        auto_prefixes=auto_prefixes,
        auto_enabled=(opts.auto_prefixes == "on"),
    )
    meta["prefixes_emitted"] = len(prefixes)

    content_blocks: list[str] = []

    default_plan = plans.get(None)
    if default_plan is not None:
        visible_triples, list_heads = default_plan
        content_blocks.extend(
            build_turtle_statement_blocks(visible_triples, prefixes, list_heads)
        )

    for graph_label, _triples in grouped.items():
        if graph_label is None:
            continue
        visible_triples, list_heads = plans[graph_label]
        if not visible_triples:
            continue
        graph_text = format_graph_label_turtle(graph_label, prefixes)
        body_blocks = build_turtle_statement_blocks(visible_triples, prefixes, list_heads)
        body = indent_multiline_block("\n".join(body_blocks), "    ")
        content_blocks.append(f"{graph_text} {{\n{body}\n}}")

    if prefixes:
        if lines:
            lines.append("")
        for prefix, namespace in prefixes.items():
            lines.append(f"@prefix {prefix}: {encode_iri_ref(namespace)} .")
        if content_blocks:
            lines.append("")
    elif opts.output_base is not None and content_blocks:
        lines.append("")

    lines.extend(content_blocks)

    if not lines:
        return "", meta
    return "\n".join(lines) + "\n", meta


def serialize_trig(
    quads: list[Quad],
    options: TurtleSerializationOptions | None = None,
) -> str:
    """Serialize quads to TriG text using the selected options."""
    text, _ = serialize_trig_with_meta(quads, options=options)
    return text


FORMAT_ALIASES = {
    "nt": "nt",
    "ntriples": "nt",
    "n-triples": "nt",
    "nq": "nq",
    "nquads": "nq",
    "n-quads": "nq",
    "ttl": "turtle",
    "turtle": "turtle",
    "trig": "trig",
}

EXTENSION_FORMATS = {
    ".nt": "nt",
    ".nq": "nq",
    ".nquads": "nq",
    ".ttl": "turtle",
    ".turtle": "turtle",
    ".trig": "trig",
}


def normalize_format(value: str) -> str:
    """Normalize a CLI format alias to the internal format key."""
    fmt = FORMAT_ALIASES.get(value.strip().lower())
    if fmt is None:
        raise ValueError(f"unsupported format: {value}")
    return fmt


def detect_format_from_path(path: str) -> str | None:
    """Infer the RDF format from a file extension."""
    if path == "-":
        return None
    return EXTENSION_FORMATS.get(Path(path).suffix.lower())


def read_input(path: str) -> str:
    """Read UTF-8 input text from a file or stdin."""
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def write_output(path: str, data: str) -> None:
    """Write output text to a file or stdout."""
    if path == "-":
        sys.stdout.write(data)
        return
    Path(path).write_text(data, encoding="utf-8")


def guess_base_iri(path: str, explicit_base: str | None) -> str | None:
    """Derive the Turtle/TriG base IRI from CLI arguments and input path."""
    if explicit_base is not None:
        return explicit_base
    if path == "-":
        return None
    return Path(path).resolve().as_uri()


def collect_node_metrics(
    node: Node,
    iri_values: set[str],
    bnode_labels: set[str],
    literal_values: set[tuple[str, str | None, str | None, str | None]],
    triple_terms: set[TripleTerm],
) -> None:
    """Accumulate unique-node statistics for one RDF node recursively."""
    if isinstance(node, IRI):
        iri_values.add(node.value)
        return
    if isinstance(node, BNode):
        bnode_labels.add(node.label)
        return
    if isinstance(node, Literal):
        literal_values.add((node.value, node.lang, node.direction, node.datatype))
        if node.datatype is not None:
            iri_values.add(node.datatype)
        return
    if isinstance(node, TripleTerm):
        triple_terms.add(node)
        collect_node_metrics(
            node.subject, iri_values, bnode_labels, literal_values, triple_terms
        )
        collect_node_metrics(
            node.predicate, iri_values, bnode_labels, literal_values, triple_terms
        )
        collect_node_metrics(
            node.object, iri_values, bnode_labels, literal_values, triple_terms
        )
        return
    raise TypeError(f"unsupported node type: {type(node)!r}")


def compute_dataset_stats(quads: list[Quad]) -> dict[str, int]:
    """Compute dataset-level counts for parsed RDF quads."""
    iri_values: set[str] = set()
    bnode_labels: set[str] = set()
    literal_values: set[tuple[str, str | None, str | None, str | None]] = set()
    triple_terms: set[TripleTerm] = set()

    subject_values: set[IRI | BNode] = set()
    predicate_values: set[IRI] = set()
    object_values: set[Node] = set()
    graph_values: set[GraphLabel] = set()
    named_graph_triples = 0
    default_graph_triples = 0

    for subject, predicate, obj, graph_label in quads:
        subject_values.add(subject)
        predicate_values.add(predicate)
        object_values.add(obj)
        collect_node_metrics(
            subject, iri_values, bnode_labels, literal_values, triple_terms
        )
        collect_node_metrics(
            predicate, iri_values, bnode_labels, literal_values, triple_terms
        )
        collect_node_metrics(
            obj, iri_values, bnode_labels, literal_values, triple_terms
        )
        if graph_label is None:
            default_graph_triples += 1
        else:
            named_graph_triples += 1
            graph_values.add(graph_label)
            collect_node_metrics(
                graph_label, iri_values, bnode_labels, literal_values, triple_terms
            )

    return {
        "triples": len(quads),
        "quads": len(quads),
        "subjects_unique": len(subject_values),
        "predicates_unique": len(predicate_values),
        "objects_unique": len(object_values),
        "named_graphs_unique": len(graph_values),
        "default_graph_triples": default_graph_triples,
        "named_graph_triples": named_graph_triples,
        "iris_unique": len(iri_values),
        "blank_nodes_unique": len(bnode_labels),
        "literals_unique": len(literal_values),
        "triple_terms_unique": len(triple_terms),
    }


def compute_graph_stats(triples: list[Triple]) -> dict[str, int]:
    """Compute graph-level counts for the parsed RDF triples."""
    return compute_dataset_stats(triples_to_quads(triples))


def convert_data(
    text: str,
    source_format: str,
    target_format: str,
    source_name: str,
    base_iri: str | None,
    turtle_options: TurtleSerializationOptions | None = None,
    dataset_to_graph_policy: str = "strict",
    selected_graph: GraphLabel | None = None,
    into_graph: GraphLabel | None = None,
) -> tuple[str, list[Quad], dict[str, int]]:
    """Parse input text and serialize it to the requested target format."""
    if source_format == "nt":
        quads = triples_to_quads(
            parse_ntriples(text, source=source_name), graph_label=into_graph
        )
    elif source_format == "nq":
        quads = parse_nquads(text, source=source_name)
    elif source_format == "turtle":
        quads = triples_to_quads(
            parse_turtle(text, source=source_name, base_iri=base_iri),
            graph_label=into_graph,
        )
    elif source_format == "trig":
        quads = parse_trig(text, source=source_name, base_iri=base_iri)
    else:
        raise ValueError(f"unsupported source format: {source_format}")

    if target_format == "nt":
        triples = quads_to_graph_triples(
            quads,
            target_format="N-Triples",
            policy=dataset_to_graph_policy,
            selected_graph=selected_graph,
        )
        return serialize_ntriples(triples), quads, {}
    if target_format == "nq":
        return serialize_nquads(quads), quads, {}
    if target_format == "turtle":
        triples = quads_to_graph_triples(
            quads,
            target_format="Turtle",
            policy=dataset_to_graph_policy,
            selected_graph=selected_graph,
        )
        output_text, serialize_meta = serialize_turtle_with_meta(
            triples, options=turtle_options
        )
        return output_text, quads, serialize_meta
    if target_format == "trig":
        output_text, serialize_meta = serialize_trig_with_meta(
            quads, options=turtle_options
        )
        return output_text, quads, serialize_meta
    raise ValueError(f"unsupported target format: {target_format}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for `rdf_converter.py`."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert RDF 1.2 between N-Triples (.nt), N-Quads (.nq), "
            "Turtle (.ttl), and TriG (.trig)."
        ),
        epilog=(
            "Notes: --turtle-style/--lists/--auto-prefixes/--prefix/--output-base "
            "apply only when target format is Turtle or TriG. "
            "--validate-only parses input only and does not write output."
        ),
    )
    parser.add_argument("input", help="Input file path, or '-' for stdin.")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output file path, or '-' for stdout. Optional with --validate-only.",
    )
    parser.add_argument(
        "--from",
        dest="source_format",
        choices=sorted(FORMAT_ALIASES),
        help="Input format. If omitted, inferred from input extension.",
    )
    parser.add_argument(
        "--to",
        dest="target_format",
        choices=sorted(FORMAT_ALIASES),
        help="Output format. If omitted, inferred from output extension.",
    )
    parser.add_argument(
        "--base",
        dest="base",
        default=None,
        help="Base IRI for Turtle/TriG input parsing (default: input file URI).",
    )
    parser.add_argument(
        "--dataset-to-graph-policy",
        choices=["strict", "default-only", "select", "union"],
        default="strict",
        help=(
            "When converting dataset formats (N-Quads/TriG) to graph formats "
            "(N-Triples/Turtle), choose how named graphs are handled."
        ),
    )
    parser.add_argument(
        "--graph",
        default=None,
        help=(
            "Graph label for --dataset-to-graph-policy select "
            "(absolute IRI or _:label)."
        ),
    )
    parser.add_argument(
        "--into-graph",
        default=None,
        help=(
            "When converting N-Triples/Turtle to N-Quads/TriG, place all input "
            "triples into this named graph (absolute IRI or _:label) instead of "
            "the default graph."
        ),
    )
    parser.add_argument(
        "--turtle-style",
        choices=["readable", "minimal"],
        default="readable",
        help="Turtle/TriG output style (target Turtle/TriG only).",
    )
    parser.add_argument(
        "--lists",
        choices=["auto", "off"],
        default="auto",
        help="List compaction mode for Turtle/TriG output (target Turtle/TriG only).",
    )
    parser.add_argument(
        "--auto-prefixes",
        choices=["on", "off"],
        default="on",
        help="Enable automatic @prefix generation (target Turtle/TriG only).",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Manual prefix binding PREFIX=IRI (repeatable, target Turtle/TriG only).",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="Emit @base directive in Turtle/TriG output (target Turtle/TriG only).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Parse and validate input only; do not write output.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print graph and serialization statistics to stderr.",
    )
    return parser


def resolve_formats(args: argparse.Namespace) -> tuple[str, str | None]:
    """Resolve and validate source/target formats from CLI arguments."""
    source = normalize_format(args.source_format) if args.source_format else None
    if source is None:
        source = detect_format_from_path(args.input)
    if source is None:
        raise ValueError("could not infer input format; provide --from")

    target = normalize_format(args.target_format) if args.target_format else None
    if target is None and args.output:
        target = detect_format_from_path(args.output)

    if args.validate_only:
        return source, target

    if args.output is None:
        raise ValueError("output path is required unless --validate-only")
    if target is None:
        raise ValueError(
            "could not infer output format; provide --to or output extension"
        )
    if source == target:
        raise ValueError("source and target formats must differ")

    return source, target


def emit_stats(stats: dict[str, int]) -> None:
    """Print collected conversion and graph statistics to stderr."""
    order = [
        "triples",
        "quads",
        "named_graphs_unique",
        "default_graph_triples",
        "named_graph_triples",
        "subjects_unique",
        "predicates_unique",
        "objects_unique",
        "iris_unique",
        "blank_nodes_unique",
        "literals_unique",
        "triple_terms_unique",
        "triples_serialized",
        "quads_serialized",
        "graphs_serialized",
        "prefixes_emitted",
        "lists_compacted",
    ]
    print("stats:", file=sys.stderr)
    for key in order:
        if key in stats:
            print(f"{key}: {stats[key]}", file=sys.stderr)
    for key in sorted(stats):
        if key not in order:
            print(f"{key}: {stats[key]}", file=sys.stderr)


def main() -> int:
    """Run the `rdf_converter.py` command-line interface."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        source, target = resolve_formats(args)
        manual_prefixes = normalize_manual_prefixes(args.prefix)
        if args.output_base is not None:
            validate_iri(args.output_base, require_absolute=False, allow_empty=False)
        selected_graph = (
            parse_cli_graph_label(args.graph, option_name="--graph")
            if args.graph is not None
            else None
        )
        into_graph = (
            parse_cli_graph_label(args.into_graph, option_name="--into-graph")
            if args.into_graph is not None
            else None
        )

        if args.dataset_to_graph_policy != "select" and args.graph is not None:
            raise ValueError("--graph requires --dataset-to-graph-policy select")
        if args.dataset_to_graph_policy == "select" and args.graph is None:
            raise ValueError(
                "--dataset-to-graph-policy select requires --graph <IRI|_:label>"
            )

        turtle_options = TurtleSerializationOptions(
            style=args.turtle_style,
            lists=args.lists,
            auto_prefixes=args.auto_prefixes,
            manual_prefixes=manual_prefixes,
            output_base=args.output_base,
        )

        input_text = read_input(args.input)
        source_name = args.input if args.input != "-" else "<stdin>"
        base_iri = guess_base_iri(args.input, args.base)

        if args.validate_only:
            if args.dataset_to_graph_policy != "strict" or args.graph is not None:
                raise ValueError(
                    "--dataset-to-graph-policy/--graph require an output conversion target"
                )
            if args.into_graph is not None:
                raise ValueError("--into-graph requires an output conversion target")
            if source == "nt":
                quads = triples_to_quads(parse_ntriples(input_text, source=source_name))
            elif source == "nq":
                quads = parse_nquads(input_text, source=source_name)
            elif source == "turtle":
                quads = triples_to_quads(
                    parse_turtle(
                        input_text, source=source_name, base_iri=base_iri
                    )
                )
            elif source == "trig":
                quads = parse_trig(
                    input_text, source=source_name, base_iri=base_iri
                )
            else:
                raise ValueError(f"unsupported source format: {source}")

            if args.stats:
                emit_stats(compute_dataset_stats(quads))
            return 0

        if target is None:
            raise ValueError("missing target format")
        if args.output is None:
            raise ValueError("missing output path")

        if target not in {"nt", "turtle"} and args.dataset_to_graph_policy != "strict":
            raise ValueError(
                "--dataset-to-graph-policy requires N-Triples or Turtle output"
            )
        if target not in {"nt", "turtle"} and args.graph is not None:
            raise ValueError("--graph requires N-Triples or Turtle output")
        if args.into_graph is not None:
            if source not in {"nt", "turtle"}:
                raise ValueError("--into-graph requires N-Triples or Turtle input")
            if target not in {"nq", "trig"}:
                raise ValueError("--into-graph requires N-Quads or TriG output")

        if target not in {"turtle", "trig"}:
            uses_turtle_output_options = (
                args.turtle_style != "readable"
                or args.lists != "auto"
                or args.auto_prefixes != "on"
                or bool(manual_prefixes)
                or args.output_base is not None
            )
            if uses_turtle_output_options:
                raise ValueError(
                    "--turtle-style/--lists/--auto-prefixes/--prefix/--output-base require Turtle or TriG output"
                )

        output_text, quads, serialize_meta = convert_data(
            text=input_text,
            source_format=source,
            target_format=target,
            source_name=source_name,
            base_iri=base_iri,
            turtle_options=turtle_options if target in {"turtle", "trig"} else None,
            dataset_to_graph_policy=args.dataset_to_graph_policy,
            selected_graph=selected_graph,
            into_graph=into_graph,
        )
        write_output(args.output, output_text)

        if args.stats:
            stats = compute_dataset_stats(quads)
            stats.update(serialize_meta)
            emit_stats(stats)
        return 0
    except (ParseError, ValueError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
