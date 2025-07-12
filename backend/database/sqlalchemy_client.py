"""
SQLAlchemy Database Client for Foto Render
Handles async database connections and session management with improved pooling
"""

import logging
import asyncio
from typing import AsyncGenerator, Optional, Any, Dict, List
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload

from database_config import get_database_url, get_direct_url
from .sqlalchemy_models import Base, AiModel, Lora, LoraCompatibility, Vae, Upscaler, Sampler, SystemConfig

logger = logging.getLogger(__name__)

# Global database client instance (singleton)
_db_client_instance: Optional['DatabaseClient'] = None
_db_client_lock = asyncio.Lock()


class DatabaseClient:
    """Async SQLAlchemy database client with improved connection pooling"""
    
    def __init__(self):
        self._engine = None
        self._session_factory = None
        self._connection_lock = asyncio.Lock()
        
    async def connect(self) -> None:
        """Initialize database connection with optimized pooling"""
        async with self._connection_lock:
            if self._engine is not None:
                return
                
            # Try direct URL first (bypasses pooling issues), fallback to pooled URL
            database_url = get_direct_url() or get_database_url()
            if not database_url:
                raise ValueError("Neither DIRECT_URL nor DATABASE_URL is configured")
                
            # Convert postgres:// or postgresql:// to postgresql+asyncpg:// and clean up Supabase params
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
            elif database_url.startswith("postgresql://") and not database_url.startswith("postgresql+asyncpg://"):
                database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
                
            # Remove pgbouncer and other Supabase-specific parameters that asyncpg doesn't understand
            if "?" in database_url:
                base_url, params = database_url.split("?", 1)
                # Keep only asyncpg-compatible parameters
                valid_params: List[str] = []
                for param in params.split("&"):
                    key: str = param.split("=")[0].lower()
                    if key not in ["pgbouncer"]:  # Add other incompatible params here if needed
                        valid_params.append(param)
                if valid_params:
                    database_url = f"{base_url}?{'&'.join(valid_params)}"
                else:
                    database_url = base_url
                
            # Optimized engine configuration for better performance
            try:
                logger.info(f"Creating async engine with URL: {database_url[:50]}...")
                self._engine = create_async_engine(
                    database_url,
                    echo=False,  # Set to True for SQL debugging
                    # Aggressive pool settings to prevent exhaustion
                    pool_size=25,         # Increased significantly - more than original
                    max_overflow=25,      # Large overflow to handle spikes
                    pool_pre_ping=True,   # Test connections before use
                    pool_recycle=1800,    # Recycle connections every 30 minutes (faster)
                    pool_timeout=10,      # Shorter timeout for getting connection
                    # Connection arguments for asyncpg with aggressive timeouts
                    connect_args={
                        "command_timeout": 30,  # Shorter command timeout
                        "server_settings": {
                            "jit": "off",  # Disable JIT for faster simple queries
                            "idle_in_transaction_session_timeout": "30000",  # 30s idle timeout
                            "statement_timeout": "30000",  # 30s statement timeout
                        },
                    }
                )
                logger.info("✅ Database engine created successfully with optimized pooling")
            except Exception as e:
                logger.error(f"❌ Failed to create database engine: {e}")
                logger.error(f"   Error type: {type(e)}")
                logger.error(f"   URL format: {database_url[:80]}...")
                import traceback
                logger.error(f"   Full traceback: {traceback.format_exc()}")
                raise e
            
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,  # Manual flush control for better performance
            )
        
    async def disconnect(self) -> None:
        """Close database connection"""
        async with self._connection_lock:
            if self._engine:
                await self._engine.dispose()
                self._engine = None
                self._session_factory = None
                logger.info("🔌 Database connections closed")
            
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session context manager with timeout"""
        if not self._session_factory:
            await self.connect()
            
        if self._session_factory:  # Type guard
            async with self._session_factory() as session:
                try:
                    yield session
                    await session.commit()  # Explicit commit for performance
                except Exception:
                    await session.rollback()
                    raise
        else:
            raise RuntimeError("Failed to initialize database session factory")
                
    async def create_tables(self) -> None:
        """Create all database tables"""
        if not self._engine:
            await self.connect()
            
        if self._engine:  # Type guard
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
    async def health_check(self) -> Dict[str, Any]:
        """Check database health with timeout"""
        try:
            async with asyncio.timeout(10.0):  # 10 second timeout
                async with self.get_session() as session:
                    result = await session.execute(text("SELECT 1"))
                    row = result.fetchone()
                    return {
                        "status": "healthy",
                        "connected": row is not None,
                        "database": "postgresql"
                    }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "database": "postgresql"
            }


# Model Service Classes
class ModelService:
    """Service for AI Model operations"""
    
    def __init__(self, db_client: DatabaseClient):
        self.db = db_client
        
    async def get_all_models(self, active_only: bool = True) -> List[AiModel]:
        """Get all AI models"""
        async with self.db.get_session() as session:
            query = select(AiModel)
            if active_only:
                query = query.where(AiModel.is_active == True)
            result = await session.execute(query)
            return list(result.scalars().all())
            
    async def get_model_by_filename(self, filename: str) -> Optional[AiModel]:
        """Get model by filename"""
        async with self.db.get_session() as session:
            result = await session.execute(
                select(AiModel).where(AiModel.filename == filename)
            )
            return result.scalar_one_or_none()
            
    async def create_model(self, model_data: Dict[str, Any]) -> AiModel:
        """Create new AI model"""
        async with self.db.get_session() as session:
            model = AiModel(**model_data)
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return model
            
    async def update_model(self, model_id: str, model_data: Dict[str, Any]) -> Optional[AiModel]:
        """Update AI model"""
        async with self.db.get_session() as session:
            result = await session.execute(
                select(AiModel).where(AiModel.id == model_id)
            )
            model = result.scalar_one_or_none()
            if model:
                for key, value in model_data.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                await session.commit()
                await session.refresh(model)
            return model


class LoraService:
    """Service for LoRA operations"""
    
    def __init__(self, db_client: DatabaseClient):
        self.db = db_client
        
    async def get_all_loras(self, active_only: bool = True, compatible_with_model: Optional[str] = None) -> List[Lora]:
        """Get all LoRAs with optional filtering"""
        try:
            async with self.db.get_session() as session:
                if compatible_with_model:
                    # Use raw SQL to avoid SQLAlchemy type issues
                    logger.info(f"🔍 Finding LoRAs compatible with model: {compatible_with_model}")
                    
                    # Raw SQL query to get compatible LoRA IDs only
                    if active_only:
                        raw_sql = text("""
                            SELECT DISTINCT l.id 
                            FROM "Lora" l
                            JOIN "LoraCompatibility" lc ON l.id = lc."loraId"
                            JOIN "AiModel" m ON lc."modelId" = m.id
                            WHERE m.filename = :model_filename AND l."isActive" = true
                        """)
                    else:
                        raw_sql = text("""
                            SELECT DISTINCT l.id 
                            FROM "Lora" l
                            JOIN "LoraCompatibility" lc ON l.id = lc."loraId"
                            JOIN "AiModel" m ON lc."modelId" = m.id
                            WHERE m.filename = :model_filename
                        """)
                    
                    result = await session.execute(raw_sql, {"model_filename": compatible_with_model})
                    lora_ids = [str(row[0]) for row in result.fetchall()]  # Ensure they're strings
                    
                    if not lora_ids:
                        logger.info(f"🚫 No LoRAs compatible with model: {compatible_with_model}")
                        return []
                    
                    # Use raw SQL for the final query too to avoid SQLAlchemy type casting
                    if active_only:
                        final_sql = text("""
                            SELECT * FROM "Lora" 
                            WHERE id = ANY(:lora_ids) AND "isActive" = true
                            ORDER BY "displayName"
                        """)
                    else:
                        final_sql = text("""
                            SELECT * FROM "Lora" 
                            WHERE id = ANY(:lora_ids)
                            ORDER BY "displayName"
                        """)
                    
                    result = await session.execute(final_sql, {"lora_ids": lora_ids})
                    rows = result.fetchall()
                    
                    # Convert rows to Lora objects manually but properly
                    loras = []
                    for row in rows:
                        lora_data = dict(row._mapping)
                        lora = Lora()
                        
                        # Map database column names to SQLAlchemy attribute names
                        column_mapping = {
                            'id': 'id',
                            'filename': 'filename',
                            'displayName': 'display_name',
                            'description': 'description',
                            'category': 'category',
                            'filePath': 'file_path',
                            'fileSize': 'file_size',
                            'checksum': 'checksum',
                            'isNsfw': 'is_nsfw',
                            'isActive': 'is_active',
                            'defaultScale': 'default_scale',
                            'minScale': 'min_scale',
                            'maxScale': 'max_scale',
                            'recommendedMin': 'recommended_min',
                            'recommendedMax': 'recommended_max',
                            'usageTips': 'usage_tips',
                            'triggerWords': 'trigger_words',
                            'author': 'author',
                            'version': 'version',
                            'website': 'website',
                            'source': 'source',
                            'tags': 'tags',
                            'usageCount': 'usage_count',
                            'lastUsed': 'last_used',
                            'createdAt': 'created_at',
                            'updatedAt': 'updated_at'
                        }
                        
                        for db_col, attr_name in column_mapping.items():
                            if db_col in lora_data and hasattr(lora, attr_name):
                                setattr(lora, attr_name, lora_data[db_col])
                        
                        loras.append(lora)
                    
                    logger.info(f"🎨 Found {len(loras)} compatible LoRAs")
                    return loras
                    
                else:
                    # For non-filtered queries, use normal SQLAlchemy
                    query = select(Lora).options(
                        selectinload(Lora.compatible_models).selectinload(LoraCompatibility.model)
                    )
                    if active_only:
                        query = query.where(Lora.is_active == True)
                    
                    result = await session.execute(query)
                    loras = list(result.scalars().all())
                    logger.info(f"🎨 Standard query returned {len(loras)} LoRAs")
                    return loras
                    
        except Exception as e:
            logger.error(f"❌ LoraService.get_all_loras failed: {e}")
            logger.error(f"❌ Query parameters: active_only={active_only}, compatible_with_model={compatible_with_model}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise
            
    async def create_lora(self, lora_data: Dict[str, Any]) -> Lora:
        """Create new LoRA"""
        async with self.db.get_session() as session:
            lora = Lora(**lora_data)
            session.add(lora)
            await session.commit()
            await session.refresh(lora)
            return lora


class VaeService:
    """Service for VAE operations"""
    
    def __init__(self, db_client: DatabaseClient):
        self.db = db_client
        
    async def get_all_vaes(self, active_only: bool = True) -> List[Vae]:
        """Get all VAEs"""
        async with self.db.get_session() as session:
            query = select(Vae)
            if active_only:
                query = query.where(Vae.is_active == True)
            result = await session.execute(query)
            return list(result.scalars().all())


class UpscalerService:
    """Service for Upscaler operations"""
    
    def __init__(self, db_client: DatabaseClient):
        self.db = db_client
        
    async def get_all_upscalers(self, active_only: bool = True) -> List[Upscaler]:
        """Get all upscalers"""
        async with self.db.get_session() as session:
            query = select(Upscaler)
            if active_only:
                query = query.where(Upscaler.is_active == True)
            result = await session.execute(query)
            return list(result.scalars().all())


class SamplerService:
    """Service for Sampler operations"""
    
    def __init__(self, db_client: DatabaseClient):
        self.db = db_client
        
    async def get_all_samplers(self, active_only: bool = True) -> List[Sampler]:
        """Get all samplers"""
        async with self.db.get_session() as session:
            query = select(Sampler)
            if active_only:
                query = query.where(Sampler.is_active == True)
            result = await session.execute(query)
            return list(result.scalars().all())
            
    async def get_default_sampler(self) -> Optional[Sampler]:
        """Get default sampler"""
        async with self.db.get_session() as session:
            result = await session.execute(
                select(Sampler).where(Sampler.is_default == True)
            )
            return result.scalar_one_or_none()


class ConfigService:
    """Service for System Configuration operations"""
    
    def __init__(self, db_client: DatabaseClient):
        self.db = db_client
        
    async def get_config(self, key: str) -> Optional[str]:
        """Get configuration value"""
        async with self.db.get_session() as session:
            result = await session.execute(
                select(SystemConfig).where(SystemConfig.key == key)
            )
            config = result.scalar_one_or_none()
            return config.value if config else None
            
    async def set_config(self, key: str, value: str, config_type: str = "string", 
                        description: Optional[str] = None, category: Optional[str] = None) -> SystemConfig:
        """Set configuration value"""
        async with self.db.get_session() as session:
            # Try to find existing config
            result = await session.execute(
                select(SystemConfig).where(SystemConfig.key == key)
            )
            config = result.scalar_one_or_none()
            
            if config:
                config.value = value
                config.type = config_type
                if description:
                    config.description = description
                if category:
                    config.category = category
            else:
                config = SystemConfig(
                    key=key,
                    value=value,
                    type=config_type,
                    description=description,
                    category=category
                )
                session.add(config)
                
            await session.commit()
            await session.refresh(config)
            return config


async def get_db_client() -> DatabaseClient:
    """Get singleton database client instance"""
    global _db_client_instance
    
    async with _db_client_lock:
        if _db_client_instance is None:
            _db_client_instance = DatabaseClient()
            await _db_client_instance.connect()
            logger.info("🗄️ Created singleton database client instance")
        
        return _db_client_instance


async def close_db_client() -> None:
    """Close singleton database client"""
    global _db_client_instance
    
    async with _db_client_lock:
        if _db_client_instance:
            await _db_client_instance.disconnect()
            _db_client_instance = None
            logger.info("🔌 Closed singleton database client") 