# RLHF Book 中文译稿

本目录只保存中文译稿需要单独维护的文件：

- `chapters/`: 中文章节 Markdown。
- `metadata.yml`: 中文书名、摘要、交叉引用标签和 PDF 字体设置。
- `templates/pdf.tex`: 中文 PDF 构建所需的 Pandoc LaTeX 模板补丁。

未翻译且与英文版完全相同的静态资源不在本目录重复保存。中文构建通过 `Makefile.zh` 从 `book/` 复用这些文件，包括：

- `book/images/`
- `book/assets/`
- `book/data/`
- `book/code/`
- `book/scripts/`
- `book/templates/` 中除中文 PDF 模板以外的文件

从仓库根目录构建中文 PDF：

```sh
make -f Makefile.zh zh-pdf PDF_ENGINE_ZH=tectonic
```

如果未来需要发布完整中文网站，再按需翻译 HTML 模板、导航、library 页面、code 页面和 redirect 配置。
