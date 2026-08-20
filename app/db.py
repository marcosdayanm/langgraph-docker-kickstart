"""Small retail schema plus async database setup."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import Field, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.settings import Settings


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"  # type: ignore[reportIncompatibleVariableOverride]

    thread_id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: str = Field(default="local-user", max_length=100)
    title: str = Field(max_length=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Product(SQLModel, table=True):
    __tablename__ = "products"  # type: ignore[reportIncompatibleVariableOverride]

    product_id: UUID = Field(default_factory=uuid4, primary_key=True)
    sku: str = Field(unique=True, index=True, max_length=40)
    name: str = Field(max_length=100)
    description: str = Field(max_length=500)
    price: Decimal = Field(max_digits=10, decimal_places=2)
    stock_quantity: int = Field(ge=0)
    initial_stock_quantity: int = Field(ge=0)


class Order(SQLModel, table=True):
    __tablename__ = "orders"  # type: ignore[reportIncompatibleVariableOverride]

    order_id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: str = Field(index=True, max_length=100)
    total_amount: Decimal = Field(max_digits=10, decimal_places=2)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OrderItem(SQLModel, table=True):
    __tablename__ = "order_items"  # type: ignore[reportIncompatibleVariableOverride]

    order_item_id: UUID = Field(default_factory=uuid4, primary_key=True)
    order_id: UUID = Field(foreign_key="orders.order_id", index=True)
    product_id: UUID = Field(foreign_key="products.product_id")
    quantity: int = Field(gt=0)
    unit_price: Decimal = Field(max_digits=10, decimal_places=2)


CATALOG = (
    {
        "sku": "TRAIL-BAG-20",
        "name": "Trail Daypack 20L",
        "description": "Water-resistant daypack with a padded laptop sleeve.",
        "price": Decimal("79.00"),
        "stock_quantity": 18,
        "initial_stock_quantity": 18,
    },
    {
        "sku": "CAMP-MUG-12",
        "name": "Insulated Camp Mug",
        "description": "12 oz stainless-steel mug for hot or cold drinks.",
        "price": Decimal("24.00"),
        "stock_quantity": 32,
        "initial_stock_quantity": 32,
    },
    {
        "sku": "URBAN-LAMP-01",
        "name": "Rechargeable Lantern",
        "description": "Compact USB-C lantern with three brightness levels.",
        "price": Decimal("39.00"),
        "stock_quantity": 12,
        "initial_stock_quantity": 12,
    },
)


def create_database(settings: Settings) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(settings.sqlalchemy_url)
    sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, sessions


async def create_tables(engine: AsyncEngine) -> None:
    """Create missing application tables; this intentionally never alters existing ones."""
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)


async def seed_catalog(session_factory: async_sessionmaker[AsyncSession]) -> None:
    """Insert or refresh immutable catalog attributes without resetting sold stock."""
    catalog_rows = [Product(**item).model_dump() for item in CATALOG]
    statement = insert(Product).values(catalog_rows)
    async with session_factory() as session:
        await session.exec(
            statement.on_conflict_do_update(
                index_elements=[Product.sku],
                set_={
                    "name": statement.excluded.name,
                    "description": statement.excluded.description,
                    "price": statement.excluded.price,
                    "initial_stock_quantity": statement.excluded.initial_stock_quantity,
                },
            )
        )
        await session.commit()
