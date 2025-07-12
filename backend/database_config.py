"""
Database and R2 Configuration for Foto Render
Centralizes all configuration settings from environment variables
"""

import os
from typing import Dict, Any, Tuple, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in the backend directory
# Force fresh reload by clearing any cached values first
import os
for key in ['DIRECT_URL', 'DATABASE_URL']:
    if key in os.environ:
        del os.environ[key]

env_path = Path(__file__).parent / '.env'
load_dotenv(env_path, override=True)  # override=True forces reload

def get_database_url() -> str:
    """Get the main database URL"""
    return os.getenv("DATABASE_URL", "")

def get_direct_url() -> str:
    """Get the direct database URL for migrations and direct access"""
    return os.getenv("DIRECT_URL", "")

def get_r2_credentials() -> Dict[str, str]:
    """Get R2/S3 credentials from environment"""
    return {
        "access_key_id": os.getenv("R2_ACCESS_KEY_ID", ""),
        "secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY", ""),
        "region": os.getenv("R2_REGION", "auto")
    }

def get_r2_config() -> Dict[str, Any]:
    """Get R2 bucket configuration"""
    bucket_name = os.getenv("R2_BUCKET_NAME", "foto-render")
    custom_domain = os.getenv("R2_CUSTOM_DOMAIN", "foto.mylinkbuddy.com")
    
    return {
        "BUCKET_NAME": bucket_name,
        "ENDPOINT_URL": os.getenv("R2_ENDPOINT_URL", "https://21de0fee19145e425579eb6f7473cead.r2.cloudflarestorage.com"),
        "PUBLIC_URL_BASE": f"https://{custom_domain}",
        "CDN_URL_BASE": f"https://{custom_domain}",  # Same as public for now
        "CUSTOM_DOMAIN": custom_domain,
        "THUMBNAILS_PATH": "thumbnails",
        "THUMBNAIL_SIZE": (512, 512),
        "MAX_FILE_SIZE": 50 * 1024 * 1024,  # 50MB
        "ALLOWED_FORMATS": ["PNG", "JPEG", "WEBP"]
    }

def get_app_config() -> Dict[str, Any]:
    """Get general application configuration"""
    return {
        "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
        "SECRET_KEY": os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    }

def validate_configuration() -> Tuple[bool, List[str]]:
    """Validate that required configuration is present"""
    errors = []
    
    # Check database configuration
    if not get_database_url():
        errors.append("DATABASE_URL not set")
    
    if not get_direct_url():
        errors.append("DIRECT_URL not set")
    
    # Check R2 configuration
    r2_creds = get_r2_credentials()
    if not r2_creds["access_key_id"]:
        errors.append("R2_ACCESS_KEY_ID not set")
    
    if not r2_creds["secret_access_key"]:
        errors.append("R2_SECRET_ACCESS_KEY not set")
    
    r2_config = get_r2_config()
    if not r2_config["ENDPOINT_URL"]:
        errors.append("R2_ENDPOINT_URL not set")
    
    return len(errors) == 0, errors

if __name__ == "__main__":
    # Test configuration when run directly
    print("🔧 Testing Foto Render Configuration...")
    
    is_valid, errors = validate_configuration()
    
    if is_valid:
        print("✅ Configuration is valid!")
        print(f"   📊 Database: {get_database_url()[:50]}...")
        print(f"   📦 R2 Bucket: {get_r2_config()['BUCKET_NAME']}")
        print(f"   🌐 Custom Domain: {get_r2_config()['CUSTOM_DOMAIN']}")
    else:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   • {error}") 