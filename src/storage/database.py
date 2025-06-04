"""
Database interface for LEXICON using asyncpg for PostgreSQL.
"""

import asyncio
import asyncpg
from typing import Any, Dict, List, Optional, Union
import json

class Database:
    """PostgreSQL database interface using asyncpg"""
    
    def __init__(self, connection_string: str):
        """Initialize database with connection string"""
        self.connection_string = connection_string
        self.pool = None
        self._connection_lock = asyncio.Lock()
        
        # Collections (similar to MongoDB collections)
        self.concepts = Collection(self, "concepts")
        self.concept_access = Collection(self, "concept_access")
        self.memetic_states = Collection(self, "memetic_states")
    
    async def connect(self):
        """Create connection pool"""
        async with self._connection_lock:
            if self.pool is None:
                # Use the postgres_database_url property from settings if available
                from src.config import settings
                connection_str = settings.postgres_database_url if hasattr(settings, 'postgres_database_url') else self.connection_string
                
                self.pool = await asyncpg.create_pool(
                    dsn=connection_str,
                    min_size=5,
                    max_size=20,
                    command_timeout=60,
                    init=self._init_connection
                )
    
    async def disconnect(self):
        """Close all connections"""
        async with self._connection_lock:
            if self.pool:
                await self.pool.close()
                self.pool = None
    
    async def _init_connection(self, conn):
        """Initialize connection with JSON encoding/decoding"""
        await conn.set_type_codec(
            'jsonb',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )
    
    async def execute(self, query: str, *args, timeout: Optional[float] = None) -> str:
        """Execute a query that doesn't return rows"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)
    
    async def fetch(self, query: str, *args, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """Execute a query and return all results as dictionaries"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args, timeout=timeout)
            return [dict(row) for row in rows]
    
    async def fetch_one(self, query: str, *args, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Execute a query and return the first result as a dictionary"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args, timeout=timeout)
            return dict(row) if row else None
    
    async def fetch_val(self, query: str, *args, column: int = 0, timeout: Optional[float] = None) -> Any:
        """Execute a query and return a single value"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)
    
    async def transaction(self):
        """Start a transaction"""
        if not self.pool:
            await self.connect()
        
        return await self.pool.acquire()
    
    async def create_tables(self):
        """Create database tables if they don't exist"""
        # This is handled by the init_db.py script
        pass


class Collection:
    """MongoDB-like collection interface for PostgreSQL tables"""
    
    def __init__(self, db: Database, table_name: str):
        self.db = db
        self.table_name = table_name
    
    async def find_one(self, filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document matching the filter"""
        conditions = []
        values = []
        
        for i, (key, value) in enumerate(filter_dict.items(), 1):
            conditions.append(f"{key} = ${i}")
            values.append(value)
        
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        
        query = f"SELECT * FROM {self.table_name} WHERE {where_clause} LIMIT 1"
        return await self.db.fetch_one(query, *values)
    
    async def find(self, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find all documents matching the filter"""
        conditions = []
        values = []
        
        if filter_dict:
            for i, (key, value) in enumerate(filter_dict.items(), 1):
                conditions.append(f"{key} = ${i}")
                values.append(value)
        
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        
        query = f"SELECT * FROM {self.table_name} WHERE {where_clause}"
        return await self.db.fetch(query, *values)
    
    async def insert_one(self, document: Dict[str, Any]) -> str:
        """Insert a single document"""
        keys = list(document.keys())
        placeholders = [f"${i+1}" for i in range(len(keys))]
        
        columns = ", ".join(keys)
        values_placeholders = ", ".join(placeholders)
        
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({values_placeholders}) RETURNING id"
        
        values = [document[key] for key in keys]
        return await self.db.fetch_val(query, *values)
    
    async def update_one(self, filter_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> int:
        """Update a single document"""
        set_clauses = []
        values = []
        
        # Build SET clause
        for i, (key, value) in enumerate(update_dict.items(), 1):
            set_clauses.append(f"{key} = ${i}")
            values.append(value)
        
        # Build WHERE clause
        where_conditions = []
        for i, (key, value) in enumerate(filter_dict.items(), len(values) + 1):
            where_conditions.append(f"{key} = ${i}")
            values.append(value)
        
        set_clause = ", ".join(set_clauses)
        where_clause = " AND ".join(where_conditions)
        
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE {where_clause}"
        result = await self.db.execute(query, *values)
        
        # Parse the number of affected rows from the result
        # Example result: "UPDATE 1"
        return int(result.split()[1])
    
    async def delete_one(self, filter_dict: Dict[str, Any]) -> int:
        """Delete a single document"""
        conditions = []
        values = []
        
        for i, (key, value) in enumerate(filter_dict.items(), 1):
            conditions.append(f"{key} = ${i}")
            values.append(value)
        
        where_clause = " AND ".join(conditions)
        
        query = f"DELETE FROM {self.table_name} WHERE {where_clause} LIMIT 1"
        result = await self.db.execute(query, *values)
        
        # Parse the number of affected rows from the result
        # Example result: "DELETE 1"
        return int(result.split()[1])
    
    async def count(self, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching the filter"""
        conditions = []
        values = []
        
        if filter_dict:
            for i, (key, value) in enumerate(filter_dict.items(), 1):
                conditions.append(f"{key} = ${i}")
                values.append(value)
        
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        
        query = f"SELECT COUNT(*) FROM {self.table_name} WHERE {where_clause}"
        return await self.db.fetch_val(query, *values)
