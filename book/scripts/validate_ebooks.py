#!/usr/bin/env python3
"""Validate EPUB 3 popup-footnote citations in the generated ebooks.

Readers turn a note into a popup instead of a page jump only when the markup
satisfies a few structural properties, so this asserts them directly:

* every citation is an ``<a epub:type="noteref">`` whose href is a *bare*
  fragment -- a link carrying a path is a cross-document navigation, which is
  precisely the thing that reads as a jump rather than a popup;
* that fragment resolves, in the same document, to an
  ``<aside epub:type="footnote">`` sitting inside an ``epub:type="footnotes"``
  container;
* each aside links back to every citation occurrence that reaches it, and those
  backlinks are ``role="doc-backlink"`` and *not* noterefs (a noteref must point
  at a note; pointing one at a citation anchor leaves readers resolving
  noterefs to non-notes);
* every cited key still has a full entry in the bibliography.

It also checks the general EPUB integrity that the above rests on: documents
must be well-formed XML and internal fragment links must resolve. Both matter
because a book that is not well-formed XML loses its EPUB 3 semantics when
Amazon converts it, so the popups silently degrade back into jumps. Do not
paper over parse failures here -- that hides the very bug this guards against.

This overlaps deliberately with epubcheck (RSC-016 and RSC-012). epubcheck
remains the authority on EPUB conformance and is worth running when available;
this script is the dependency-free build-time gate.
"""

from __future__ import annotations

import argparse
import posixpath
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
EPUB_NS = "http://www.idpf.org/2007/ops"
NCX_NS = "http://www.daisy.org/z3986/2005/ncx/"
OPF_NS = "http://www.idpf.org/2007/opf"
XHTML_NS = "http://www.w3.org/1999/xhtml"

EPUB_TYPE = f"{{{EPUB_NS}}}type"


class EbookValidationError(Exception):
    """Raised when an EPUB does not satisfy the citation contract."""


@dataclass
class Document:
    path: str
    root: ElementTree.Element
    parents: dict[ElementTree.Element, ElementTree.Element]
    ids: dict[str, ElementTree.Element]


@dataclass
class Summary:
    documents: int
    identifiers: int
    noterefs: int
    footnotes: int
    backlinks: int
    bibliography_entries: int


def has_token(value: str | None, token: str) -> bool:
    return token in (value or "").split()


def local_name(element: ElementTree.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def parse_xml(content: bytes, path: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        raise EbookValidationError(
            f"{path}: not well-formed XML: {exc}\n"
            "    EPUB content documents must parse as XML. A stray '<' or '&' in "
            "an inline <style>/<script> body will break this; wrap such content "
            "in a CDATA section."
        ) from exc


def resolve(base: str, href: str) -> tuple[str, str] | None:
    """Resolve an in-archive href to (document path, fragment), or None."""
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc or not parsed.fragment:
        return None
    if parsed.path:
        document = posixpath.normpath(
            posixpath.join(posixpath.dirname(base), unquote(parsed.path))
        )
    else:
        document = base
    return document, unquote(parsed.fragment)


def package_path(archive: zipfile.ZipFile) -> str:
    try:
        container = parse_xml(
            archive.read("META-INF/container.xml"), "META-INF/container.xml"
        )
    except KeyError as exc:
        raise EbookValidationError("missing META-INF/container.xml") from exc

    rootfile = container.find(f".//{{{CONTAINER_NS}}}rootfile")
    full_path = rootfile.get("full-path") if rootfile is not None else None
    if not full_path:
        raise EbookValidationError("container.xml has no package rootfile")
    return full_path


def load(archive: zipfile.ZipFile) -> tuple[dict[str, Document], list[str]]:
    """Parse every XHTML document in the manifest, plus any NCX paths."""
    opf_path = package_path(archive)
    package = parse_xml(archive.read(opf_path), opf_path)
    base = posixpath.dirname(opf_path)

    documents: dict[str, Document] = {}
    ncx_paths: list[str] = []
    failures: list[str] = []

    for item in package.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item"):
        href = item.get("href")
        media_type = item.get("media-type")
        if not href:
            continue
        path = posixpath.normpath(
            posixpath.join(base, unquote(urlsplit(href).path))
        )
        if media_type == "application/x-dtbncx+xml":
            ncx_paths.append(path)
            continue
        if media_type != "application/xhtml+xml":
            continue

        try:
            raw = archive.read(path)
        except KeyError:
            failures.append(f"{path}: listed in manifest but missing from archive")
            continue

        try:
            root = parse_xml(raw, path)
        except EbookValidationError as exc:
            failures.append(str(exc))
            continue

        parents = {
            child: parent for parent in root.iter() for child in parent
        }
        ids: dict[str, ElementTree.Element] = {}
        duplicates: Counter[str] = Counter()
        for element in root.iter():
            identifier = element.get("id")
            if not identifier:
                continue
            if identifier in ids:
                duplicates[identifier] += 1
            ids[identifier] = element
        for identifier, count in duplicates.items():
            failures.append(
                f"{path}: duplicate id {identifier!r} ({count + 1} occurrences)"
            )

        documents[path] = Document(path=path, root=root, parents=parents, ids=ids)

    # A parse failure cascades into spurious unresolved-fragment noise, so stop.
    if failures:
        raise EbookValidationError("\n".join(failures))
    if not documents:
        raise EbookValidationError("package manifest contains no XHTML documents")
    return documents, ncx_paths


def check_fragment_links(
    documents: dict[str, Document],
    archive: zipfile.ZipFile,
    ncx_paths: list[str],
) -> list[str]:
    """Every internal fragment link must resolve. (epubcheck RSC-012.)"""
    errors: list[str] = []

    def check(source: str, href: str, where: str) -> None:
        resolved = resolve(source, href)
        if resolved is None:
            return
        target_path, fragment = resolved
        target = documents.get(target_path)
        if target is None:
            errors.append(f"{where}: link to unknown document {target_path!r}")
        elif fragment not in target.ids:
            errors.append(
                f"{where}: {href!r} does not resolve "
                f"(no id {fragment!r} in {target_path})"
            )

    for path, document in documents.items():
        for link in document.root.iter(f"{{{XHTML_NS}}}a"):
            href = link.get("href")
            if href:
                check(path, href, path)

    for ncx_path in ncx_paths:
        try:
            ncx = parse_xml(archive.read(ncx_path), ncx_path)
        except (KeyError, EbookValidationError) as exc:
            errors.append(f"{ncx_path}: unreadable ({exc})")
            continue
        for content in ncx.iter(f"{{{NCX_NS}}}content"):
            src = content.get("src")
            if src:
                check(ncx_path, src, ncx_path)

    return errors


def check_citations(documents: dict[str, Document]) -> tuple[Summary, list[str]]:
    errors: list[str] = []
    noterefs = 0
    footnotes = 0
    backlinks = 0

    # Bibliography entries citeproc emitted, keyed by CSL id.
    bibliography: set[str] = set()
    for document in documents.values():
        for element in document.root.iter():
            identifier = element.get("id") or ""
            if identifier.startswith("ref-") and has_token(
                element.get("class"), "csl-entry"
            ):
                bibliography.add(identifier)

    # Which citation anchors point at which note, per document.
    sources_by_note: dict[tuple[str, str], list[str]] = defaultdict(list)

    for path, document in documents.items():
        for link in document.root.iter(f"{{{XHTML_NS}}}a"):
            epub_type = link.get(EPUB_TYPE)
            href = link.get("href") or ""

            if has_token(link.get("class"), "citation-backlink"):
                backlinks += 1
                if has_token(epub_type, "noteref"):
                    errors.append(
                        f"{path}: backlink {href!r} is marked "
                        "epub:type=noteref; a noteref must point at a note"
                    )
                if not has_token(link.get("role"), "doc-backlink"):
                    errors.append(
                        f"{path}: backlink {href!r} lacks role=doc-backlink"
                    )
                if not href.startswith("#"):
                    errors.append(
                        f"{path}: backlink {href!r} is not a bare fragment"
                    )
                continue

            if not has_token(epub_type, "noteref"):
                continue

            noterefs += 1
            source_id = link.get("id")
            if not source_id:
                errors.append(f"{path}: noteref {href!r} has no id to link back to")
            if not href.startswith("#"):
                errors.append(
                    f"{path}: noteref {href!r} is not a bare fragment; a "
                    "cross-document note is navigated to, not popped up"
                )
                continue

            fragment = unquote(href[1:])
            note = document.ids.get(fragment)
            if note is None:
                errors.append(f"{path}: noteref {href!r} has no target in document")
                continue
            if local_name(note) != "aside" or not has_token(
                note.get(EPUB_TYPE), "footnote"
            ):
                errors.append(
                    f"{path}: noteref {href!r} targets "
                    f"<{local_name(note)} epub:type={note.get(EPUB_TYPE)!r}>; "
                    "expected <aside epub:type='footnote'>"
                )
                continue
            if source_id:
                sources_by_note[(path, fragment)].append(source_id)

    for path, document in documents.items():
        for note in document.root.iter(f"{{{XHTML_NS}}}aside"):
            if not has_token(note.get(EPUB_TYPE), "footnote"):
                continue
            footnotes += 1
            note_id = note.get("id")
            if not note_id:
                errors.append(f"{path}: footnote aside has no id")
                continue

            # The container carries the footnotes semantic that readers look for.
            container = document.parents.get(note)
            if container is None or not has_token(
                container.get(EPUB_TYPE), "footnotes"
            ):
                errors.append(
                    f"{path}: footnote #{note_id} is not inside an "
                    "epub:type='footnotes' container"
                )

            target = note.get("data-citation-target")
            if target and target not in bibliography:
                errors.append(
                    f"{path}: footnote #{note_id} copies {target!r}, "
                    "which has no bibliography entry"
                )

            expected = sources_by_note.get((path, note_id), [])
            if not expected:
                errors.append(
                    f"{path}: footnote #{note_id} is unreachable from any noteref"
                )
                continue

            # Pandoc's native footnotes (plain markdown [^1] notes) are already
            # in the right shape but carry no backlinks; only our cloned
            # citation entries promise them.
            if not has_token(note.get("class"), "citation-footnote"):
                continue

            found = {
                unquote((link.get("href") or "")[1:])
                for link in note.iter(f"{{{XHTML_NS}}}a")
                if has_token(link.get("class"), "citation-backlink")
            }
            for source_id in expected:
                if source_id not in found:
                    errors.append(
                        f"{path}: footnote #{note_id} has no backlink to "
                        f"citation #{source_id}"
                    )
                elif source_id not in document.ids:
                    errors.append(
                        f"{path}: footnote #{note_id} links back to "
                        f"#{source_id}, which does not exist"
                    )

    if noterefs == 0:
        errors.append("no citation noterefs found; the citation filter did not run")
    if footnotes == 0:
        errors.append("no footnote asides found; the citation filter did not run")

    summary = Summary(
        documents=len(documents),
        identifiers=sum(len(d.ids) for d in documents.values()),
        noterefs=noterefs,
        footnotes=footnotes,
        backlinks=backlinks,
        bibliography_entries=len(bibliography),
    )
    return summary, errors


def validate(path: Path) -> Summary:
    if not path.is_file():
        raise EbookValidationError(f"{path}: not a file")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise EbookValidationError(f"{path}: not a valid ZIP archive: {exc}") from exc

    with archive:
        broken = archive.testzip()
        if broken is not None:
            raise EbookValidationError(f"{path}: corrupt archive entry {broken}")

        documents, ncx_paths = load(archive)
        errors = check_fragment_links(documents, archive, ncx_paths)
        summary, citation_errors = check_citations(documents)
        errors.extend(citation_errors)

    if errors:
        shown = errors[:25]
        suffix = (
            f"\n  ... and {len(errors) - len(shown)} more" if len(errors) > len(shown) else ""
        )
        raise EbookValidationError(
            "\n".join(f"  {error}" for error in shown) + suffix
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("epubs", nargs="+", type=Path, help="EPUB files to validate")
    return parser.parse_args()


def main() -> int:
    status = 0
    for path in parse_args().epubs:
        try:
            summary = validate(path)
        except EbookValidationError as exc:
            print(f"FAIL {path}\n{exc}", file=sys.stderr)
            status = 1
        else:
            print(
                f"OK   {path}: {summary.documents} documents, "
                f"{summary.identifiers} ids, {summary.noterefs} noterefs -> "
                f"{summary.footnotes} popup footnotes, "
                f"{summary.backlinks} backlinks, "
                f"{summary.bibliography_entries} bibliography entries"
            )
    return status


if __name__ == "__main__":
    sys.exit(main())
