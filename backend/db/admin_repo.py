from motor.motor_asyncio import AsyncIOMotorClient

from config import settings
from models import Admin, AdminResponse

_client = AsyncIOMotorClient(settings.mongodb_url)
_col = _client[settings.mongodb_db]["admins"]


async def upsert_admin(name: str, email: str, photo_url: str = "") -> AdminResponse:
    """Create a new admin or update an existing admin profile by email."""
    existing = await _col.find_one({"email": email})

    if existing:
      await _col.update_one(
          {"email": email},
          {"$set": {"name": name, "photo_url": photo_url, "role": "admin"}},
      )
      updated = await _col.find_one({"email": email}, {"_id": 0})
      return AdminResponse(**updated)

    admin = Admin(
        name=name,
        email=email,
        photo_url=photo_url,
    )
    await _col.insert_one(admin.dict())
    return AdminResponse(**admin.dict())


async def get_admin_by_email(email: str) -> AdminResponse | None:
    """Find an admin by email."""
    doc = await _col.find_one({"email": email}, {"_id": 0})
    if doc:
        return AdminResponse(**doc)
    return None
