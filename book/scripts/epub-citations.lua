-- Give citeproc citations EPUB 3 popup-footnote semantics.
--
-- Kindle (and Apple Books, Thorium, ...) render a note as a popup instead of
-- navigating to it only when the markup matches the shape readers already
-- recognise from native footnotes, which is what pandoc's own EPUB 3 writer
-- emits:
--
--   <a href="#n1" class="footnote-ref" epub:type="noteref" role="doc-noteref">1</a>
--   ...
--   <section class="footnotes" epub:type="footnotes">
--   <hr />
--   <aside id="n1" epub:type="footnote" role="doc-footnote"><p>note text</p></aside>
--   </section>
--
-- Three properties of that shape are load-bearing:
--
--   1. the note lives in the *same* XHTML document as the reference, reached by
--      a bare fragment (`#n1`), because a cross-document link is a navigation,
--      not a popup;
--   2. the notes sit inside a container marked `epub:type="footnotes"`;
--   3. the note body is a single flow of text in a `<p>`.
--
-- Pandoc splits the EPUB into one document per level-1 header, so this filter
-- clones each cited bibliography entry into a per-chapter <aside>. A reference
-- cited several times in one chapter shares one aside and collects one backlink
-- per occurrence. The full bibliography at the back of the book is left alone.
--
-- Backlinks are plain links carrying role="doc-backlink" only. Marking them as
-- noteref (an earlier attempt at "bidirectional" linking) is wrong: a noteref
-- must point at a note, and these point at the citation anchor, which leaves a
-- reader resolving noterefs to non-notes.
--
-- Must run after citeproc, so citation links and bibliography entries already
-- carry their generated #ref-* identifiers.

local bibliography_entries = {}
local citation_sources = {}
local known_ids = {}
local source_count = 0
local note_count = 0
local section_count = 0

-- Citations reachable only through an image's alt text must stay untouched: the
-- writers stringify alt text, so a noteref there would be silently dropped.
local image_alt_marker = "data-epub-citation-alt"

local chapter_notes = {}
local chapter_notes_by_target = {}

-- CSL wraps entry pieces in layout spans that only make sense next to the
-- hanging indent of a real bibliography, not inside a popup.
local csl_layout_classes = {
  ["csl-block"] = true,
  ["csl-indent"] = true,
  ["csl-left-margin"] = true,
  ["csl-right-inline"] = true,
}

-- Pandoc rewrites every internal link in the AST to a document-qualified href
-- when it splits the EPUB (even a same-document one becomes "ch002.xhtml#id"),
-- which reads to a reader as a cross-document navigation. Its own native
-- footnotes keep a bare "#fn1" because the writer generates them after that
-- rewriting. Raw HTML is the only way for a filter to get the same result, so
-- the anchors below are emitted raw with their text kept as AST so pandoc still
-- escapes it.
local function xml_escape(value)
  return value
    :gsub("&", "&amp;")
    :gsub('"', "&quot;")
    :gsub("<", "&lt;")
    :gsub(">", "&gt;")
end

local function has_class(element, class_name)
  for _, class in ipairs(element.classes) do
    if class == class_name then
      return true
    end
  end
  return false
end

local function is_csl_layout_span(inline)
  if inline.t ~= "Span" then
    return false
  end
  for _, class in ipairs(inline.classes) do
    if csl_layout_classes[class] then
      return true
    end
  end
  return false
end

local function remember_id(element)
  if element.identifier and element.identifier ~= "" then
    known_ids[element.identifier] = true
  end
  return nil
end

local function collect_ids(doc)
  local attr_elements = {
    "Cell",
    "Code",
    "CodeBlock",
    "Div",
    "Figure",
    "Header",
    "Image",
    "Link",
    "Row",
    "Span",
    "Table",
  }
  local filter = {}
  for _, element_name in ipairs(attr_elements) do
    filter[element_name] = remember_id
  end
  doc:walk(filter)
end

local function unique_id(prefix, counter)
  local candidate
  repeat
    counter = counter + 1
    candidate = prefix .. counter
  until not known_ids[candidate]
  known_ids[candidate] = true
  return candidate, counter
end

local function next_source_id()
  local id
  id, source_count = unique_id("cite-", source_count)
  return id
end

local function next_note_id()
  local id
  id, note_count = unique_id("cite-note-", note_count)
  return id
end

local function next_section_id()
  local id
  id, section_count = unique_id("citation-footnotes-", section_count)
  return id
end

local function remember_bibliography_entry(entry)
  if not has_class(entry, "csl-entry")
      or not entry.identifier:match("^ref%-.+$") then
    return nil
  end
  if bibliography_entries[entry.identifier] then
    error("duplicate bibliography entry: #" .. entry.identifier)
  end
  bibliography_entries[entry.identifier] = entry:clone()
  return nil
end

local function mark_image_alt_links(image)
  return image:walk({
    Link = function(link)
      if link.target:match("^#ref%-.+$") then
        link.attributes[image_alt_marker] = "true"
      end
      return link
    end,
  })
end

-- Turn a citation link into a noteref pointing at this chapter's copy of the
-- entry, registering the occurrence so the aside can link back to it.
local function mark_noteref(link)
  if link.attributes[image_alt_marker] then
    link.attributes[image_alt_marker] = nil
    return link
  end

  local target = link.target:match("^#(ref%-.+)$")
  if not target then
    return nil
  end

  local note = chapter_notes_by_target[target]
  if not note then
    note = {
      identifier = next_note_id(),
      target = target,
      source_ids = {},
    }
    chapter_notes_by_target[target] = note
    table.insert(chapter_notes, note)
  end

  local source_id = next_source_id()
  table.insert(note.source_ids, source_id)
  citation_sources[target] = citation_sources[target] or {}
  table.insert(citation_sources[target], source_id)

  local inlines = pandoc.List({
    pandoc.RawInline("html", string.format(
      '<a id="%s" href="#%s" class="footnote-ref"'
        .. ' epub:type="noteref" role="doc-noteref">',
      xml_escape(source_id),
      xml_escape(note.identifier)
    )),
  })
  inlines:extend(link.content)
  inlines:insert(pandoc.RawInline("html", "</a>"))
  return inlines
end

-- Flatten a bibliography entry into one run of inlines, dropping the CSL
-- layout spans but keeping their text (including the "[31]" label).
local function entry_inlines(entry)
  local flattened = pandoc.List()

  local function absorb(inlines)
    for _, inline in ipairs(inlines) do
      if is_csl_layout_span(inline) then
        absorb(inline.content)
      else
        flattened:insert(inline)
      end
    end
  end

  absorb(pandoc.utils.blocks_to_inlines(entry.content, { pandoc.Space() }))
  return flattened
end

local function backlink_inlines(source_ids)
  local inlines = pandoc.List()
  for index, source_id in ipairs(source_ids) do
    inlines:insert(pandoc.Space())
    inlines:insert(pandoc.RawInline("html", string.format(
      '<a href="#%s" class="citation-backlink" role="doc-backlink"'
        .. ' aria-label="Back to citation %d">',
      xml_escape(source_id),
      index
    )))
    inlines:insert(pandoc.Str("\u{21A9}"))
    if #source_ids > 1 then
      inlines:insert(pandoc.Superscript({ pandoc.Str(tostring(index)) }))
    end
    inlines:insert(pandoc.RawInline("html", "</a>"))
  end
  return inlines
end

local function note_aside(note)
  local entry = bibliography_entries[note.target]
  if not entry then
    return nil
  end

  local content = entry_inlines(entry:clone())
  content:extend(backlink_inlines(note.source_ids))

  return {
    pandoc.RawBlock("html", string.format(
      '<aside id="%s" class="citation-footnote"'
        .. ' data-citation-target="%s" epub:type="footnote"'
        .. ' role="doc-footnote">',
      xml_escape(note.identifier),
      xml_escape(note.target)
    )),
    pandoc.Para(content),
    pandoc.RawBlock("html", "</aside>"),
  }
end

-- Emit the notes gathered for the chapter that just ended.
local function finish_chapter(output)
  local asides = pandoc.List()
  for _, note in ipairs(chapter_notes) do
    local aside = note_aside(note)
    if aside then
      asides:extend(aside)
    end
  end

  if #asides > 0 then
    output:insert(pandoc.RawBlock("html", string.format(
      '<section id="%s" class="footnotes citation-footnotes"'
        .. ' epub:type="footnotes">',
      xml_escape(next_section_id())
    )))
    output:insert(pandoc.HorizontalRule())
    output:extend(asides)
    output:insert(pandoc.RawBlock("html", "</section>"))
  end

  chapter_notes = {}
  chapter_notes_by_target = {}
end

local function localize_citations(blocks)
  local output = pandoc.Blocks({})
  chapter_notes = {}
  chapter_notes_by_target = {}

  for _, block in ipairs(blocks) do
    if block.t == "Header" and block.level == 1 then
      finish_chapter(output)
    end
    output:insert(block:walk({ Link = mark_noteref }))
  end
  finish_chapter(output)

  return output
end

local function check_targets()
  local unresolved = {}
  for target, _ in pairs(citation_sources) do
    if not bibliography_entries[target] then
      table.insert(unresolved, target)
    end
  end
  table.sort(unresolved)

  if #unresolved > 0 then
    error(
      "unresolved citation target(s): #" .. table.concat(unresolved, ", #")
    )
  end
end

function Pandoc(doc)
  collect_ids(doc)
  doc:walk({ Div = remember_bibliography_entry })
  doc = doc:walk({ Image = mark_image_alt_links })
  doc.blocks = localize_citations(doc.blocks)
  check_targets()
  return doc
end
