local json = pandoc.json
local stringify = pandoc.utils.stringify

local function meta_text(meta, key, default)
  local value = meta[key]
  if value == nil then
    return default
  end

  local text = stringify(value)
  if text == "" then
    return default
  end
  return text
end

local function json_block(value)
  return pandoc.MetaBlocks({ pandoc.RawBlock("html", json.encode(value)) })
end

function Meta(meta)
  local lang = meta_text(meta, "lang", "en-US")
  local canonical_url = meta_text(meta, "canonical-url", "https://rlhfbook.com/")
  local default_site_description =
    "A free online book and course on RLHF, preference tuning, reward models, RLVR, and post-training language models."
  local default_chapter_description =
    "A chapter from the RLHF Book, a free guide to reinforcement learning from human feedback and post-training language models."

  local book_description = meta_text(meta, "description", default_site_description)
  local chapter_title = meta_text(meta, "page-title", meta_text(meta, "title", "RLHF Book"))
  local chapter_description = meta_text(meta, "meta-description", default_chapter_description)

  meta["book-jsonld"] = json_block({
    ["@context"] = "https://schema.org",
    ["@type"] = "Book",
    name = "Reinforcement Learning from Human Feedback",
    alternateName = "The RLHF Book",
    description = book_description,
    author = {
      ["@type"] = "Person",
      name = "Nathan Lambert",
    },
    url = canonical_url,
    image = "https://rlhfbook.com/assets/rlhf-book-cover.png",
    inLanguage = lang,
    isAccessibleForFree = true,
    keywords = "RLHF, post-training, language models, reward models, preference tuning, DPO, RLVR",
    sameAs = {
      "https://github.com/natolambert/rlhf-book",
      "https://arxiv.org/abs/2504.12501",
      "https://www.manning.com/books/the-rlhf-book",
    },
  })

  meta["chapter-jsonld"] = json_block({
    ["@context"] = "https://schema.org",
    ["@type"] = "Chapter",
    headline = chapter_title,
    description = chapter_description,
    url = canonical_url,
    image = "https://rlhfbook.com/assets/rlhf-book-share.png",
    author = {
      ["@type"] = "Person",
      name = "Nathan Lambert",
    },
    isPartOf = {
      ["@type"] = "Book",
      name = "Reinforcement Learning from Human Feedback",
      url = "https://rlhfbook.com/",
    },
    inLanguage = lang,
  })

  meta["breadcrumb-jsonld"] = json_block({
    ["@context"] = "https://schema.org",
    ["@type"] = "BreadcrumbList",
    itemListElement = {
      {
        ["@type"] = "ListItem",
        position = 1,
        name = "RLHF Book",
        item = "https://rlhfbook.com/",
      },
      {
        ["@type"] = "ListItem",
        position = 2,
        name = chapter_title,
        item = canonical_url,
      },
    },
  })

  return meta
end
