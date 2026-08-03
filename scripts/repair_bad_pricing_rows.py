"""One-off repair for catalogue rows written with unit-error prices.

Before pricing_service._select_representative, colliding LiteLLM listings were
resolved last-key-wins, letting wandb's per-million unit errors price shared
models at up to 540,000x their real rate -- and event_service bills off these
rows as 'database-exact'.

Read-only by default: lists rows above MAX_PRICE_PER_1K and the events priced
against them. With --apply, deactivates those rows so the next sync rewrites
them. Event costs are deliberately NOT restated -- the right replacement price
depends on which host the customer used, which the event does not record.

    python -m scripts.repair_bad_pricing_rows           # inspect (default)
    python -m scripts.repair_bad_pricing_rows --apply   # deactivate bad rows

Point DATABASE_URL at the target first. Take a backup before --apply.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, or_, select, update  # noqa: E402

from app.common import MAX_PRICE_PER_1K  # noqa: E402
from app.database import async_session_maker  # noqa: E402
from app.models.db_models import Event, ModelPricing  # noqa: E402


async def main(apply: bool) -> int:
    async with async_session_maker() as db:
        bad = (await db.execute(
            select(ModelPricing).where(or_(
                ModelPricing.input_price_per_1k > MAX_PRICE_PER_1K,
                ModelPricing.output_price_per_1k > MAX_PRICE_PER_1K,
            )).order_by(ModelPricing.output_price_per_1k.desc())
        )).scalars().all()

        if not bad:
            print(f"No rows above ${MAX_PRICE_PER_1K}/1k. Nothing to repair.")
            return 0

        print(f"{len(bad)} row(s) above ${MAX_PRICE_PER_1K}/1k:\n")
        print(f"  {'model_name':<52} {'in/1k':>12} {'out/1k':>12} {'provider':<12}")
        names = []
        for row in bad:
            names.append(row.model_name)
            print(f"  {row.model_name[:52]:<52} {row.input_price_per_1k:>12.4f} "
                  f"{row.output_price_per_1k:>12.4f} {row.provider:<12}")

        # Blast radius: events that were costed against these names.
        impact = (await db.execute(
            select(
                Event.model,
                func.count(Event.id),
                func.sum(Event.cost),
            )
            .where(Event.model.in_(names))
            .group_by(Event.model)
        )).all()

        print()
        if not impact:
            print("No stored events reference these models. Catalogue-only problem.")
        else:
            print("STORED EVENTS PRICED AGAINST THESE ROWS -- review before restating:")
            print(f"  {'model':<52} {'events':>8} {'total cost':>16}")
            total_events = total_cost = 0
            for model, count, cost in impact:
                total_events += count
                total_cost += float(cost or 0)
                print(f"  {model[:52]:<52} {count:>8} {float(cost or 0):>16.4f}")
            print(f"  {'TOTAL':<52} {total_events:>8} {total_cost:>16.4f}")

        if not apply:
            print("\nDry run. Re-run with --apply to deactivate the rows above.")
            return 0

        result = await db.execute(
            update(ModelPricing)
            .where(ModelPricing.model_name.in_(names))
            # The marker lets the next sync reactivate these rows with clean
            # prices; without it they would stay retired forever.
            .values(is_active=False,
                    notes="auto-deactivated: unit-error price (repair_bad_pricing_rows)")
        )
        await db.commit()
        print(f"\nDeactivated {result.rowcount} row(s). "
              "Run a pricing sync to repopulate them with corrected prices.")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="deactivate the offending rows (default: dry run)")
    raise SystemExit(asyncio.run(main(parser.parse_args().apply)))
