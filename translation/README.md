# 中文术语维护

本目录用于维护 RLHF Book 中文译本的术语决策。

- `TERMS.zh.tsv` 是机器可处理的术语表，适合后续批量替换、审校和统计。
- `../TRANSLATION_GLOSSARY.zh.md` 是给译者阅读的 Markdown 版本，适合快速查阅。

## 推荐流程

如果要把某个译法全书统一替换，先 dry-run：

```sh
python3 scripts/replace_translation_term.py --old "偏好调优" --new "偏好微调"
```

确认命中位置合理后再执行：

```sh
python3 scripts/replace_translation_term.py --old "偏好调优" --new "偏好微调" --apply
```

默认替换范围包括：

- `book-zh/chapters/*.md`
- `book-zh/metadata.yml`

脚本会跳过 Markdown fenced code blocks，避免改动代码示例。若术语表本身也需要同步更新，可加：

```sh
python3 scripts/replace_translation_term.py --old "偏好调优" --new "偏好微调" --include-term-files --apply
```

执行替换后，建议运行：

```sh
python3 scripts/check_translation_integrity.py
make -f Makefile.zh zh-pdf PDF_ENGINE_ZH=tectonic
```

## 字段说明

`TERMS.zh.tsv` 使用制表符分隔，字段如下：

- `english`: 英文术语。
- `zh`: 当前推荐中文译法。
- `first_mention`: 首次出现时建议写法。
- `keep_english`: 是否通常保留英文或缩写。
- `category`: 术语类别。
- `alternatives`: 可接受或曾使用过的其他译法。
- `notes`: 使用说明。
