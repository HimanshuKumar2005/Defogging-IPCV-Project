"""Streamlit entry point for the image dehazing comparison application.

The user interface will be added after the project scaffold is established.
"""

from __future__ import annotations


def main() -> None:
    """Run the application placeholder."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is required to run the application. "
            "Install dependencies with `python -m pip install -r requirements.txt`."
        ) from exc

    st.set_page_config(page_title="Image Dehazing", layout="wide")
    st.title("Image Dehazing: Classical Image Processing vs Deep Learning")
    st.info("The interactive comparison interface will be implemented next.")


if __name__ == "__main__":
    main()
