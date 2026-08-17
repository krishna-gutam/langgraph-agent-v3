"""Launcher for the Streamlit frontend: `streamlit run main.py`.

`main()` must be *called* here, not merely imported. Streamlit re-executes this
file on every interaction, but Python caches imported modules - so a frontend
that rendered at import time would draw the page once and then blank it on the
first click.
"""

from frontends.streamlit_app import main

main()
