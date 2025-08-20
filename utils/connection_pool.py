# utils/connection_pool.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ETL_scripts.loader import WRDSLoader
from queue import Queue

from contextlib import contextmanager


class ConnectionPool:
    def __init__(self, username, max_connections=5):
        self.username = username
        self.pool = Queue(maxsize=max_connections)
        self.active_connections = 0
        self.max_connections = max_connections

    def get_connection(self):
        if self.pool.empty() and self.active_connections < self.max_connections:
            # Create new connection
            connection = WRDSLoader(username=self.username)
            self.active_connections += 1
        else:
            # Get existing connection from pool
            connection = self.pool.get()
        return connection

    @contextmanager
    def connection(self):
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.pool.put(conn)