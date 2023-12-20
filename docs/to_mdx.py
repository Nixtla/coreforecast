import re
import shutil
from pathlib import Path

comment_pat = re.compile(r"<!--.*?-->", re.DOTALL)
anchor_pat = re.compile(r"<a.*?>(.*?)</a>")
output_path = Path("docs/mintlify")

# process docs
for file in Path("docs").glob("*.md"):
    text = file.read_text()
    text = comment_pat.sub("", text)
    text = anchor_pat.sub("", text)
    module_name = file.name.split(".")[-2]
    output_file = output_path / (module_name + ".mdx")
    output_file.write_text(text)

# copy readme
shutil.copy("README.md", output_path / "index.mdx")
