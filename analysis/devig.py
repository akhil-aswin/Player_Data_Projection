"""
Odds math: convert American odds to implied probability, strip out the
bookmaker's vig (overround), and build a no-vig consensus line across
multiple sportsbooks for the same player prop.
"""

from dataclasses import dataclass


def american_to_prob(odds: float) -> float:
    """Convert American odds to raw (vig-included) implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def prob_to_american(prob: float) -> float:
    """Convert a probability back to American odds, for display purposes."""
    if prob >= 0.5:
        return -100 * prob / (1 - prob)
    return 100 * (1 - prob) / prob


@dataclass
class DevigResult:
    fair_prob_over: float
    fair_prob_under: float
    vig_pct: float


def devig_two_way(over_odds: float, under_odds: float) -> DevigResult:
    """
    Multiplicative de-vig: divides out the bookmaker's overround
    proportionally. This is the standard method for near-even two-way
    markets like player props (point spread / totals-style de-vig methods
    like Shin's are more relevant for lopsided moneylines).
    """
    p_over_raw = american_to_prob(over_odds)
    p_under_raw = american_to_prob(under_odds)
    overround = p_over_raw + p_under_raw

    fair_over = p_over_raw / overround
    fair_under = p_under_raw / overround
    vig_pct = (overround - 1) * 100

    return DevigResult(fair_prob_over=fair_over, fair_prob_under=fair_under, vig_pct=vig_pct)


def consensus_fair_probability(book_lines: list) -> dict:
    """
    Takes odds from multiple books for the SAME player/stat and returns a
    consensus no-vig probability.

    book_lines: [{"book": str, "line": float, "over_odds": int, "under_odds": int}, ...]

    Books are grouped by the numeric line they're quoting (books occasionally
    disagree, e.g. 24.5 vs 25.5 — that disagreement is itself worth noticing,
    so it's flagged in the output rather than silently averaged away).
    """
    if not book_lines:
        raise ValueError("No book lines provided")

    devigged = []
    for bl in book_lines:
        result = devig_two_way(bl["over_odds"], bl["under_odds"])
        devigged.append({**bl, "fair_prob_over": result.fair_prob_over, "vig_pct": result.vig_pct})

    lines_seen = {bl["line"] for bl in book_lines}
    # use whichever line the majority of books are quoting
    consensus_line = max(lines_seen, key=lambda ln: sum(1 for bl in book_lines if bl["line"] == ln))
    matching = [d for d in devigged if d["line"] == consensus_line]

    avg_fair_prob_over = sum(d["fair_prob_over"] for d in matching) / len(matching)
    avg_vig_pct = sum(d["vig_pct"] for d in matching) / len(matching)

    return {
        "consensus_line": consensus_line,
        "fair_prob_over": avg_fair_prob_over,
        "fair_prob_under": 1 - avg_fair_prob_over,
        "avg_vig_pct": avg_vig_pct,
        "n_books": len(matching),
        "lines_disagreement": len(lines_seen) > 1,
        "all_lines_seen": sorted(lines_seen),
        "per_book": devigged,
    }
