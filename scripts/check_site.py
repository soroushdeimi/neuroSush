"""Check built site pages using only the Python standard library."""

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from neurosush.viz.export import available_plates


class LandingParser(HTMLParser):
    """Check nesting, duplicate IDs, local links, automatic resource URLs, and anchor nesting."""

    def __init__(self, root: Path, current_file: Path):
        super().__init__(convert_charrefs=True)
        self.root = root
        self.current_file = current_file
        self.stack: list[str] = []
        self.ids: set[str] = set()
        self.fragments: list[str] = []
        self.anchor_depth = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "a":
            line, _col = self.getpos()
            assert self.anchor_depth == 0, (
                f"Nested anchor tag <a> found in "
                f"{self.current_file.relative_to(self.root)} at line {line}"
            )
            self.anchor_depth += 1
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
        if tag == "a":
            self.anchor_depth -= 1

    def handle_endtag(self, tag):
        if tag == "a":
            self.anchor_depth -= 1
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

    # Verify atlas.json against available_plates
    atlas_path = args.out / "assets" / "atlas.json"
    assert atlas_path.is_file(), f"Missing atlas.json at {atlas_path}"
    atlas_data = json.loads(atlas_path.read_text(encoding="utf-8"))
    shipped_plates = {p for p in available_plates() if p != "demo"}

    atlas_item_ids = {item["id"] for item in atlas_data["items"]}
    for p in shipped_plates:
        assert p in atlas_item_ids, f"Shipped plate {p} missing from atlas.json"

    plate_ids_in_atlas = [
        item["id"]
        for item in atlas_data["items"]
        if item["id"] in shipped_plates or item["id"].startswith(("M", "N", "H", "T", "Y", "P"))
    ]
    assert len(plate_ids_in_atlas) == len(set(plate_ids_in_atlas)), (
        "Duplicate plate ID found in atlas.json"
    )

    for pid in plate_ids_in_atlas:
        if pid.startswith(("M", "N", "H", "T", "Y", "P")):
            assert pid in shipped_plates, f"Atlas.json names a plate that is not shipped: {pid}"

    print(
        "Site HTML nesting, IDs, local links, local assets, anchor nesting, and atlas.json: passed"
    )


if __name__ == "__main__":
    main()
