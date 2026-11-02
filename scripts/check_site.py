"""Check built site pages using only the Python standard library."""

import argparse
import re
from html.parser import HTMLParser
from pathlib import Path


class LandingParser(HTMLParser):
    """Check nesting, duplicate IDs, local links, and automatic resource URLs."""

    def __init__(self, root: Path, current_file: Path):
        super().__init__(convert_charrefs=True)
        self.root = root
        self.current_file = current_file
        self.stack: list[str] = []
        self.ids: set[str] = set()
        self.fragments: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag not in {"meta", "link", "input", "br", "hr", "img", "source", "wbr"}:
            self.stack.append(tag)
        if "id" in attributes:
            identifier = attributes["id"]
            assert identifier not in self.ids, f"Duplicate ID: {identifier}"
            self.ids.add(identifier)
        for name in ("src", "href"):
            url = attributes.get(name)
            if not url:
                continue
            if url.startswith("#"):
                if len(url) > 1:
                    self.fragments.append(url[1:])
            elif url.startswith(("https:", "http:", "//", "mailto:")):
                assert tag == "a", f"External resource: {url}"
            else:
                clean_url = url.split("?")[0].split("#")[0]
                if clean_url:
                    target = (self.current_file.parent / clean_url).resolve()
                    assert target.is_file(), f"Missing file: {url} (resolved to {target})"

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()

    def handle_endtag(self, tag):
        assert self.stack, f"Unexpected closing tag: {tag}"
        assert self.stack[-1] == tag, f"Mismatched closing tag: {tag}"
        self.stack.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="Built site directory")
    args = parser.parse_args()
    for path in sorted(args.out.glob("*.html")):
        page = path.read_text(encoding="utf-8")
        check = LandingParser(args.out, path)
        check.feed(page)
        check.close()
        assert not check.stack, f"Unclosed tags in {path.relative_to(args.out)}: {check.stack}"
        assert all(fragment in check.ids for fragment in check.fragments), (
            f"Broken fragment in {path.relative_to(args.out)}"
        )
        assert "{{" not in page, f"Unrendered template token in {path.relative_to(args.out)}"
    for name in ("site.css", "site.js"):
        text = (args.out / "assets" / name).read_text(encoding="utf-8")
        assert not re.search(r"(?:https?:)?//[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    print("Site HTML nesting, IDs, local links, and local assets: passed")


if __name__ == "__main__":
    main()
