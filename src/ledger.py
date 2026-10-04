"""
Volumetric Mining Auditor (VMA) - Cryptographic Ledger & Provenance Module
Implements an append-only SQLite audit ledger with chained SHA-256 digests
and dynamic verification QR code generation.
"""

from __future__ import annotations
import base64
import hashlib
import io
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import qrcode


GENESIS_PREV_HASH = "0" * 64


def compute_sha256(data: Union[str, bytes, np.ndarray, Path]) -> str:
    """
    Compute hex-encoded SHA-256 checksum over string, bytes, numpy array, or file path.
    """
    hasher = hashlib.sha256()

    if isinstance(data, Path) or (isinstance(data, str) and os.path.isfile(data)):
        with open(data, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    elif isinstance(data, np.ndarray):
        hasher.update(data.tobytes())
    elif isinstance(data, str):
        hasher.update(data.encode("utf-8"))
    elif isinstance(data, bytes):
        hasher.update(data)
    else:
        hasher.update(str(data).encode("utf-8"))

    return hasher.hexdigest()


class AuditLedger:
    """
    Append-only SQLite ledger recording chained cryptographic provenance records
    for volumetric audits.
    """

    def __init__(self, db_path: str = "audit_ledger.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_records (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    pre_dem_hash TEXT NOT NULL,
                    post_dem_hash TEXT NOT NULL,
                    lease_geojson TEXT NOT NULL,
                    lease_hash TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    current_hash TEXT NOT NULL UNIQUE
                );
                """
            )
            conn.commit()

    def get_latest_hash(self) -> str:
        """Fetch the most recent block's hash, or the genesis hash if table is empty."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT current_hash FROM audit_records ORDER BY record_id DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                return str(row["current_hash"])
        return GENESIS_PREV_HASH

    def append_record(
        self,
        pre_dem_hash: str,
        post_dem_hash: str,
        lease_geojson: Union[str, Dict[str, Any]],
        metrics: Dict[str, Any],
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Append a new tamper-evident audit record to the ledger.
        
        Args:
            pre_dem_hash: SHA-256 digest of pre-excavation DEM raster.
            post_dem_hash: SHA-256 digest of post-excavation DEM raster.
            lease_geojson: GeoJSON geometry string or dictionary of legal lease boundary.
            metrics: Dictionary of computed volumetric audit metrics.
            timestamp: Optional ISO-8601 timestamp (defaults to UTC now).
            
        Returns:
            Dictionary representation of the stored audit record.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        if isinstance(lease_geojson, dict):
            lease_geojson_str = json.dumps(lease_geojson, sort_keys=True)
        else:
            lease_geojson_str = str(lease_geojson)

        lease_hash = compute_sha256(lease_geojson_str)
        metrics_json_str = json.dumps(metrics, sort_keys=True)

        previous_hash = self.get_latest_hash()

        # Deterministic payload construction
        payload = (
            f"{previous_hash}|"
            f"{pre_dem_hash}|"
            f"{post_dem_hash}|"
            f"{lease_hash}|"
            f"{metrics_json_str}|"
            f"{timestamp}"
        )
        current_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO audit_records (
                    timestamp,
                    previous_hash,
                    pre_dem_hash,
                    post_dem_hash,
                    lease_geojson,
                    lease_hash,
                    metrics_json,
                    current_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    previous_hash,
                    pre_dem_hash,
                    post_dem_hash,
                    lease_geojson_str,
                    lease_hash,
                    metrics_json_str,
                    current_hash,
                ),
            )
            record_id = cur.lastrowid
            conn.commit()

        return {
            "record_id": record_id,
            "timestamp": timestamp,
            "previous_hash": previous_hash,
            "pre_dem_hash": pre_dem_hash,
            "post_dem_hash": post_dem_hash,
            "lease_geojson": lease_geojson_str,
            "lease_hash": lease_hash,
            "metrics": metrics,
            "current_hash": current_hash,
        }

    def verify_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify cryptographic integrity of the entire ledger chain.
        Returns:
            (is_valid: bool, errors: List[str])
        """
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM audit_records ORDER BY record_id ASC")
            records = cur.fetchall()

        if not records:
            return True, []

        errors = []
        expected_prev = GENESIS_PREV_HASH

        for row in records:
            rec_id = row["record_id"]
            if row["previous_hash"] != expected_prev:
                errors.append(
                    f"Record {rec_id}: previous_hash mismatch! Expected {expected_prev}, "
                    f"got {row['previous_hash']}"
                )

            # Check lease hash integrity
            expected_lease_hash = compute_sha256(row["lease_geojson"])
            if row["lease_hash"] != expected_lease_hash:
                errors.append(f"Record {rec_id}: lease_hash integrity check failed!")

            # Check current hash integrity
            payload = (
                f"{row['previous_hash']}|"
                f"{row['pre_dem_hash']}|"
                f"{row['post_dem_hash']}|"
                f"{row['lease_hash']}|"
                f"{row['metrics_json']}|"
                f"{row['timestamp']}"
            )
            computed_current = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if row["current_hash"] != computed_current:
                errors.append(
                    f"Record {rec_id}: current_hash mismatch! Tampered payload detected."
                )

            expected_prev = row["current_hash"]

        return (len(errors) == 0, errors)

    def get_records(self) -> List[Dict[str, Any]]:
        """Return all audit records stored in the ledger."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM audit_records ORDER BY record_id ASC")
            rows = cur.fetchall()
            return [
                {
                    "record_id": r["record_id"],
                    "timestamp": r["timestamp"],
                    "previous_hash": r["previous_hash"],
                    "pre_dem_hash": r["pre_dem_hash"],
                    "post_dem_hash": r["post_dem_hash"],
                    "lease_hash": r["lease_hash"],
                    "lease_geojson": r["lease_geojson"],
                    "metrics": json.loads(r["metrics_json"]),
                    "current_hash": r["current_hash"],
                }
                for r in rows
            ]

    def get_latest_record(self) -> Optional[Dict[str, Any]]:
        records = self.get_records()
        return records[-1] if records else None


def generate_verification_qr(
    audit_hash: str,
    metadata: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> Image.Image:
    """
    Generate a cryptographic verification QR code certificate image encoding
    the SHA-256 audit digest and compliance summary.
    
    Args:
        audit_hash: 64-character hex SHA-256 hash of the audit record.
        metadata: Optional dictionary with volume, compliance status, timestamp, etc.
        output_path: Optional file path to save the generated PNG image.
        
    Returns:
        PIL.Image.Image: High-resolution QR code image.
    """
    certificate_payload: Dict[str, Any] = {
        "title": "Volumetric Mining Auditor - Verified Audit Seal",
        "protocol": "VMA-SHA256-v1.0",
        "audit_hash": audit_hash,
        "verification_uri": f"https://vma-audit.gov/verify/{audit_hash}",
    }
    if metadata:
        certificate_payload["metadata"] = metadata

    payload_json = json.dumps(certificate_payload, indent=2, sort_keys=True)

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=3,
    )
    qr.add_data(payload_json)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    pil_img = img.get_image() if hasattr(img, "get_image") else img

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(output_path)

    return pil_img


def qr_to_base64(img: Image.Image) -> str:
    """Convert PIL QR image to base64 data URI string for direct web embedding."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64_str}"
