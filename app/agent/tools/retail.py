"""Small ORM-backed catalog and checkout tools."""

from __future__ import annotations

import json
from decimal import Decimal

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from langgraph.types import interrupt
from sqlalchemy import desc
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import Order, OrderItem, Product


def retail_tools(session_factory: async_sessionmaker[AsyncSession]) -> list[BaseTool]:
    """Create catalog lookup, checkout, and private order-history tools."""

    @tool
    async def find_articles(name: str = "") -> str:
        """Find up to 10 products, optionally filtered by a case-insensitive name match.

        Results include SKU and stock for agent decisions. Do not volunteer either to
        customers: use name, description, and price when recommending products. State
        the stock number only when the customer explicitly asks for the purchase limit.
        """
        search_term = name.strip()
        async with session_factory() as session:
            statement = select(Product).order_by(col(Product.name)).limit(10)
            if search_term:
                statement = statement.where(col(Product.name).ilike(f"%{search_term}%"))
            products = (await session.exec(statement)).all()
        return json.dumps(
            [
                {
                    "sku": product.sku,
                    "name": product.name,
                    "description": product.description,
                    "price": str(product.price),
                    "stock_quantity": product.stock_quantity,
                }
                for product in products
            ]
        )

    @tool
    async def create_order(sku: str, quantity: int, runtime: ToolRuntime) -> str:
        """After customer approval, create one order and decrement available inventory."""
        if quantity < 1:
            return "Quantity must be at least 1."
        approved = interrupt(
            {
                "kind": "approval",
                "action": f"Place an order for {quantity} unit(s) of {sku}.",
            }
        )
        if approved is not True:
            return "The customer did not approve the order."
        user_id = getattr(runtime.context, "user_id", "local-user")
        async with session_factory() as session:
            product = (
                await session.exec(select(Product).where(col(Product.sku) == sku).with_for_update())
            ).one_or_none()
            if product is None:
                return "That product is no longer available."
            if product.stock_quantity < quantity:
                return "There is not enough inventory for that purchase."
            total = product.price * Decimal(quantity)
            order = Order(user_id=user_id, total_amount=total)
            item = OrderItem(
                order_id=order.order_id,
                product_id=product.product_id,
                quantity=quantity,
                unit_price=product.price,
            )
            product.stock_quantity -= quantity
            session.add(order)
            session.add(item)
            await session.commit()
        return f"Order {order.order_id} created for {product.name}; total ${total:.2f}."

    @tool
    async def get_my_recent_orders(runtime: ToolRuntime) -> str:
        """Return the current customer's five latest retail orders and purchased items."""
        user_id = getattr(runtime.context, "user_id", "local-user")
        async with session_factory() as session:
            rows = (
                await session.exec(
                    select(Order, OrderItem, Product)
                    .join(OrderItem, col(OrderItem.order_id) == col(Order.order_id))
                    .join(Product, col(Product.product_id) == col(OrderItem.product_id))
                    .where(col(Order.user_id) == user_id)
                    .order_by(desc(col(Order.created_at)))
                    .limit(5)
                )
            ).all()
        return json.dumps(
            [
                {
                    "order_id": str(order.order_id),
                    "created_at": order.created_at.isoformat(),
                    "product": product.name,
                    "sku": product.sku,
                    "quantity": item.quantity,
                    "total": str(order.total_amount),
                }
                for order, item, product in rows
            ]
        )

    return [find_articles, create_order, get_my_recent_orders]
