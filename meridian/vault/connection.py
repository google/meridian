"""Utility functions for interacting with the Meridian Vault PostgreSQL database.

This module intentionally stops short of defining a full schema for model results.
It focuses on connection management and provides helpers that will be fleshed out
once the precise database contract is known.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence

import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import execute_values

# Environment variable defaults -------------------------------------------------
# These variables can be overridden to point at any PostgreSQL instance.
DEFAULT_DB_NAME = os.environ.get("MERIDIAN_VAULT_DB_NAME", "meridian")
DEFAULT_DB_USER = os.environ.get("MERIDIAN_VAULT_DB_USER", "meridian")
DEFAULT_DB_PASSWORD = os.environ.get("MERIDIAN_VAULT_DB_PASSWORD", "meridian")
DEFAULT_DB_HOST = os.environ.get("MERIDIAN_VAULT_DB_HOST", "localhost")
DEFAULT_DB_PORT = int(os.environ.get("MERIDIAN_VAULT_DB_PORT", "5432"))
DEFAULT_MODEL_RESULTS_TABLE = os.environ.get(
    "MERIDIAN_VAULT_MODEL_RESULTS_TABLE", "model_results"
)


def get_connection_params(
    overrides: Optional[Mapping[str, object]] = None,
) -> MutableMapping[str, object]:
    """Build the keyword arguments used when connecting to PostgreSQL.

    Parameters
    ----------
    overrides:
        Optional mapping of connection keyword arguments that should override the
        environment driven defaults. Only keys with non-``None`` values are
        applied to the returned dictionary.
    """

    params: MutableMapping[str, object] = {
        "dbname": DEFAULT_DB_NAME,
        "user": DEFAULT_DB_USER,
        "password": DEFAULT_DB_PASSWORD,
        "host": DEFAULT_DB_HOST,
        "port": DEFAULT_DB_PORT,
    }

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                params[key] = value

    return params


@contextmanager
def get_connection(
    overrides: Optional[Mapping[str, object]] = None,
) -> Iterator[PgConnection]:
    """Yield a PostgreSQL connection configured for the vault database."""

    connection = psycopg2.connect(**get_connection_params(overrides))
    try:
        yield connection
    finally:
        connection.close()


def initialize_schema(
    connection: Optional[PgConnection] = None,
    table_name: Optional[str] = None,
    overrides: Optional[Mapping[str, object]] = None,
) -> None:
    """Ensure that a placeholder results table exists.

    The actual schema for model results will be finalized later. For now we
    create a minimal table that stores the raw model payload as JSON for
    prototyping purposes.
    """

    table = table_name or DEFAULT_MODEL_RESULTS_TABLE
    statement = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id SERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        payload JSONB NOT NULL
        -- TODO: Add concrete model output fields once they are defined.
    );
    """

    if connection is not None:
        with connection.cursor() as cursor:
            cursor.execute(statement)
        connection.commit()
        return

    with get_connection(overrides) as managed_connection:
        initialize_schema(
            connection=managed_connection,
            table_name=table,
        )


def send_model_results(
    results: Iterable[Mapping[str, object]],
    connection: Optional[PgConnection] = None,
    table_name: Optional[str] = None,
    overrides: Optional[Mapping[str, object]] = None,
) -> None:
    """Persist raw model result payloads in the vault database.

    Parameters
    ----------
    results:
        Iterable of dictionaries representing model outputs. Every dictionary will
        be serialized to JSON and stored in the ``payload`` column until a more
        detailed schema is available.
    connection:
        Optional existing PostgreSQL connection. If omitted, a new connection is
        created for the duration of the call.
    table_name:
        Destination table name. Defaults to ``MERIDIAN_VAULT_MODEL_RESULTS_TABLE``.
    overrides:
        Optional overrides passed to :func:`get_connection` when an implicit
        connection is required.
    """

    payloads = list(results)
    if not payloads:
        return

    table = table_name or DEFAULT_MODEL_RESULTS_TABLE

    if connection is not None:
        initialize_schema(connection=connection, table_name=table)
        _insert_payloads(connection, table, payloads)
        return

    with get_connection(overrides) as managed_connection:
        initialize_schema(connection=managed_connection, table_name=table)
        _insert_payloads(managed_connection, table, payloads)


def _insert_payloads(
    connection: PgConnection,
    table_name: str,
    payloads: Sequence[Mapping[str, object]],
) -> None:
    """Insert serialized payloads into the database."""

    serialized_values = [(json.dumps(payload),) for payload in payloads]
    with connection.cursor() as cursor:
        execute_values(
            cursor,
            f"INSERT INTO {table_name} (payload) VALUES %s",
            serialized_values,
        )
    connection.commit()
