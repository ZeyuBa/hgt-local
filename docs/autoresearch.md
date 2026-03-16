# Autoresearch Skill 说明

`autoresearch` 已重构为标准 Skill 目录：`skills/autoresearch-hgt/`。

## 新的规范位置
- Skill 主入口：`skills/autoresearch-hgt/SKILL.md`
- 参考文档：`skills/autoresearch-hgt/references/`
- 工具脚本：`skills/autoresearch-hgt/scripts/`
- 会话模板：`skills/autoresearch-hgt/assets/`

## 运行期产物存储规范
实验期间的可变文件（`progress.md`、`results.tsv`、`session.log`）不应放在 `docs/`。
统一放到：

```text
outputs/research/<run-tag>/
```

`outputs/` 已被 `.gitignore` 忽略，避免把临时实验记录混入仓库版本历史。
