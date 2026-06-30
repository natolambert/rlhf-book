# 中文术语维护

本目录用于维护 RLHF Book 中文译本的术语决策。

- `TERMS.zh.tsv` 是机器可处理的术语表，适合后续批量替换、审校和统计。
- `../TRANSLATION_GLOSSARY.zh.md` 是给译者阅读的 Markdown 版本，适合快速查阅。

## 推荐流程

术语脚本只做“精确匹配候选查找”和“确认后的机械替换”。它不能判断一个词在具体上下文中是否应该改，也不能发现所有语义等价但表面写法不同的译法。因此推荐流程是：

1. 先更新或确认 `TERMS.zh.tsv` 中的术语决策。
2. 用 dry-run 导出候选及上下文，交给人工或 LLM 审阅。
3. 只在候选合理时执行 `--apply`。
4. 如果存在多个旧译法，分别对每个旧译法运行一次。

例如，要把“偏好调优”统一改成“偏好微调”，先查看命中数量：

```sh
python3 scripts/replace_translation_term.py --old "偏好调优" --new "偏好微调"
```

如果数量不小，导出上下文供人工或 LLM 审阅：

```sh
python3 scripts/replace_translation_term.py \
  --old "偏好调优" \
  --new "偏好微调" \
  --context 2 \
  --review-file tmp/term-review.tsv
```

确认命中位置合理后再执行机械替换：

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

如果一次术语调整不是简单替换，例如要区分 “alignment” 在不同上下文中译为“对齐”还是保留 “AI alignment”，不要直接 `--apply`。先用 `rg` 搜索相关英文和中文候选词，再把候选上下文合并审阅。

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
