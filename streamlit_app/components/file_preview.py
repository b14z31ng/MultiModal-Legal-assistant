"""File preview component for uploaded documents."""

from __future__ import annotations

from typing import Optional

import streamlit as st
from PIL import Image
import io


def render_file_preview(
    file_bytes: bytes,
    filename: str,
) -> None:
    """Render a preview of the uploaded file.

    Displays image preview for image files and a
    file info card for non-previewable types.

    Args:
        file_bytes: Raw file content.
        filename: Name of the file.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    image_exts = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}

    if ext in image_exts:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            st.image(
                image,
                caption=filename,
                use_container_width=True,
            )
        except Exception:
            _render_file_info(filename, len(file_bytes), ext)
    elif ext == "pdf":
        _render_file_info(filename, len(file_bytes), ext, icon="📄")
        st.caption(
            "PDF preview not available. "
            "The document will be processed via OCR."
        )
    elif ext == "docx":
        _render_file_info(filename, len(file_bytes), ext, icon="📝")
        st.caption(
            "DOCX preview not available. "
            "Text will be extracted directly."
        )
    else:
        _render_file_info(filename, len(file_bytes), ext)


def _render_file_info(
    filename: str,
    size_bytes: int,
    extension: str,
    icon: str = "📎",
) -> None:
    """Render file information card.

    Args:
        filename: File name.
        size_bytes: File size in bytes.
        extension: File extension.
        icon: Emoji icon for the file type.
    """
    size_str = _format_file_size(size_bytes)

    html = f"""
    <div style="
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    ">
        <div style="font-size: 3rem; margin-bottom: 0.75rem;">{icon}</div>
        <div style="
            color: #f1f5f9;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
            word-break: break-all;
        ">{filename}</div>
        <div style="color: #94a3b8; font-size: 0.85rem;">
            {extension.upper()} • {size_str}
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable units.

    Args:
        size_bytes: File size in bytes.

    Returns:
        Formatted size string (e.g., '1.5 MB').
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
