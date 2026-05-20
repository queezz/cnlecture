"""
mkdocs-macros module for cnlecture.

Defines the ``base_url`` variable so that iframe src attributes in markdown
pages can reference site-root-relative assets portably:

    <iframe src="{{ base_url }}/assets/plots/euler_spiral.html">

``base_url`` is the relative path from the current page back to the site
root (e.g. ``../..`` for a page two directories deep).  This mirrors the
``base_url`` variable available in MkDocs Jinja2 theme templates.
"""


def define_env(env):
    """Entry-point called once when the plugin initialises."""
    # Sensible default; overwritten per-page in on_pre_page_macros.
    env.variables["base_url"] = "."


def on_pre_page_macros(env):
    """Recompute base_url before macros are expanded on each page.

    env.page.url is the root-relative URL of the page, e.g.
    'examples/euler_spiral/' for a page two levels deep.
    We count path segments and build a relative '../..' path accordingly.
    """
    url = env.page.url.strip("/")
    depth = url.count("/") + 1 if url else 0
    env.variables["base_url"] = "/".join([".."] * depth) if depth else "."
