python3 - << 'PYEOF'
path = "depa_downstream/config_schema.py"
with open(path) as f:
    content = f.read()

old = 'VALID_CONTRAST_TYPES = ["pairwise", "one_vs_rest"]'
new = 'VALID_CONTRAST_TYPES = ["pairwise", "one_vs_rest", "auto_one_vs_rest_all"]'

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("config_schema.py 已更新：VALID_CONTRAST_TYPES 加入 auto_one_vs_rest_all")
elif "auto_one_vs_rest_all" in content:
    print("已经是最新版，无需修改")
else:
    print("⚠ 没找到预期的旧字符串，可能文件结构变了，请手动检查")
PYEOF
