# connectivity_postgres.py
import asyncio
import logging
from types import FunctionType
from app.core import database  # or from app.core.database import initialize_pool, execute_query, close_pool

logger = logging.getLogger(__name__)

async def maybe_await(func, *args, **kwargs):
    """Call func; if it returns a coroutine, await it; otherwise return result."""
    res = func(*args, **kwargs)
    if asyncio.iscoroutine(res):
        return await res
    return res

async def check_connection_async():
    try:
        # initialize_pool might be sync or async
        await maybe_await(database.initialize_pool)

        # execute_query might be sync or async; it may return dict directly
        result = await maybe_await(database.execute_query, "SELECT 1 AS ok", fetch_one=True)
        if result and result.get("ok") == 1:
            print("✅ Successfully connected to the database via Prisma!")
        else:
            print("❌ Unexpected response from database:", result)

    except Exception as exc:
        print("❌ Failed to connect to the database:")
        print(exc)
    finally:
        # call close_pool and await it if it returns a coroutine
        try:
            await maybe_await(database.close_pool)
        except Exception as exc:
            # if close_pool misbehaves (e.g. schedules a coroutine but doesn't return it),
            # try a second-best attempt to cleanly disconnect Prisma if available
            print("Warning: close_pool raised an error:", exc)
            # best-effort: attempt to call prisma.disconnect if exported
            prisma = getattr(database, "prisma", None)
            if prisma is not None:
                # if prisma.disconnect is coroutine, await it
                disc = prisma.disconnect()
                if asyncio.iscoroutine(disc):
                    await disc

def check_connection():
    asyncio.run(check_connection_async())

if __name__ == "__main__":
    check_connection()

