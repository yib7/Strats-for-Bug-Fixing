"""What the published documentation site is allowed to load.

The site is served from GitHub Pages, which lets you set no response headers of your own --
what arrives is `Strict-Transport-Security` plus GitHub's defaults, and there is no way to
add a `Content-Security-Policy` from a `_headers` file or a config knob. So the only place
"this site loads nothing from a third party" can be enforced is at build time, here.

That claim is load-bearing: `docs/index.md`'s network-and-telemetry section tells readers the
site loads no third-party scripts, fonts or stylesheets, and cycle 6 had to remove exactly such
a dependency -- the readthedocs theme's default `highlightjs: true`, which pulls highlight.js
and its stylesheet from `cdnjs.cloudflare.com` on every page, unpinned and without Subresource
Integrity, handing a visitor's IP and User-Agent to Cloudflare. `mkdocs.yml` turns it off and
highlights with Pygments at build time instead. Nothing stops a future theme bump from turning
it back on, which is what these tests are for.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Any absolute or protocol-relative URL in a `src=`/`href=` attribute.
_ASSET_URL = re.compile(r"""(?:src|href)\s*=\s*["'](//[^"']+|https?://[^"']+)["']""")

# Attributes that name a *link* rather than something the browser fetches while rendering.
# A hyperlink to GitHub is fine; a stylesheet, script, font or image is not.
_LINK_RELS_THAT_DO_NOT_FETCH = ("canonical", "alternate", "prev", "next")


@pytest.fixture(scope="module")
def built_site(tmp_path_factory) -> Path:
    """`mkdocs build --strict` into a tmp dir.

    Never into the repo's `site/`: that would make running the suite mutate the working tree,
    and CI's clean-tree check exists precisely to catch that.
    """
    mkdocs = pytest.importorskip("mkdocs.commands.build")
    config_mod = pytest.importorskip("mkdocs.config")

    out = tmp_path_factory.mktemp("site")
    cfg = config_mod.load_config(str(REPO_ROOT / "mkdocs.yml"), site_dir=str(out), strict=True)
    cfg.plugins.on_startup(command="build", dirty=False)
    mkdocs.build(cfg)
    return out


def _fetched_urls(html: str) -> list[str]:
    """Absolute/protocol-relative URLs the browser would fetch to render `html`."""
    found = []
    for tag in re.findall(r"<(?:script|link|img|source|iframe|audio|video)\b[^>]*>", html):
        if any(f'rel="{rel}"' in tag for rel in _LINK_RELS_THAT_DO_NOT_FETCH):
            continue
        found.extend(_ASSET_URL.findall(tag))
    return found


def test_no_page_fetches_a_third_party_asset(built_site):
    """Every script, stylesheet, font and image must be same-origin."""
    offenders: dict[str, list[str]] = {}
    pages = sorted(built_site.rglob("*.html"))
    assert pages, "mkdocs produced no HTML -- the build fixture is broken, not the site"

    for page in pages:
        urls = _fetched_urls(page.read_text(encoding="utf-8", errors="replace"))
        if urls:
            offenders[str(page.relative_to(built_site))] = urls

    assert not offenders, (
        f"the site would fetch third-party assets: {offenders}. "
        "Vendor them into docs/ instead; the site is documented as loading nothing external."
    )


def test_the_highlightjs_cdn_pull_stays_off(built_site):
    """The specific regression cycle 6 removed, pinned by name."""
    for page in built_site.rglob("*.html"):
        text = page.read_text(encoding="utf-8", errors="replace")
        assert "cdnjs.cloudflare.com" not in text, f"{page.name} pulls from cdnjs"
        assert "highlight.min.js" not in text, f"{page.name} pulls highlight.js"


def test_nothing_is_loaded_over_plain_http(built_site):
    """Mixed content: the site is served over HTTPS, so an `http://` asset would be blocked
    by the browser anyway -- and would have leaked the request in the clear before that."""
    for page in built_site.rglob("*.html"):
        for url in _fetched_urls(page.read_text(encoding="utf-8", errors="replace")):
            assert not url.startswith("http://"), f"{page.name} loads {url} over plain HTTP"


def test_no_analytics_or_tracker_is_configured():
    """`mkdocs.yml` must not grow an analytics block, and no page may embed a tracker."""
    import yaml

    config = yaml.safe_load((REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    assert "google_analytics" not in config
    assert "analytics" not in (config.get("extra") or {})

    trackers = (
        "googletagmanager",
        "google-analytics",
        "gtag(",
        "plausible.io",
        "matomo",
        "hotjar",
        "mixpanel",
        "segment.com",
    )
    for page in sorted((REPO_ROOT / "docs").rglob("*.md")):
        text = page.read_text(encoding="utf-8", errors="replace").lower()
        for tracker in trackers:
            assert tracker not in text, f"{page.name} mentions {tracker}"
