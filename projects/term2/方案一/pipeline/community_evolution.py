from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set, Tuple


def jaccard_similarity(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def build_month_community_members(
    assignment_rows: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[int, Set[str]]]:
    by_month: Dict[str, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in assignment_rows:
        month = str(row.get("feature_month", "")).strip()
        global_id = str(row.get("global_id", "")).strip()
        community_id = row.get("community_id")
        if not month or not global_id or community_id is None:
            continue
        by_month[month][int(community_id)].add(global_id)
    return {month: dict(community_map) for month, community_map in by_month.items()}


def track_community_evolution(assignment_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    month_members = build_month_community_members(assignment_rows)
    months = sorted(month_members.keys())
    if len(months) < 2:
        return []

    output_rows: List[Dict[str, Any]] = []
    for from_month, to_month in zip(months, months[1:]):
        from_map = month_members[from_month]
        to_map = month_members[to_month]
        matches: List[Dict[str, Any]] = []
        for from_community_id, from_members in sorted(from_map.items()):
            best_to_community_id = None
            best_jaccard = 0.0
            for to_community_id, to_members in to_map.items():
                score = jaccard_similarity(from_members, to_members)
                if score > best_jaccard:
                    best_jaccard = score
                    best_to_community_id = to_community_id
            matches.append(
                {
                    "from_community_id": int(from_community_id),
                    "to_community_id": None if best_to_community_id is None else int(best_to_community_id),
                    "from_size": len(from_members),
                    "to_size": len(to_map.get(best_to_community_id, set())) if best_to_community_id is not None else 0,
                    "jaccard": best_jaccard,
                }
            )

        from_assignment = {
            global_id: community_id
            for community_id, members in from_map.items()
            for global_id in members
        }
        to_assignment = {
            global_id: community_id
            for community_id, members in to_map.items()
            for global_id in members
        }
        migrations: List[Dict[str, Any]] = []
        for global_id in sorted(set(from_assignment) & set(to_assignment)):
            if from_assignment[global_id] != to_assignment[global_id]:
                migrations.append(
                    {
                        "global_id": global_id,
                        "from_community_id": int(from_assignment[global_id]),
                        "to_community_id": int(to_assignment[global_id]),
                    }
                )

        output_rows.append(
            {
                "from_month": from_month,
                "to_month": to_month,
                "community_matches": matches,
                "company_migrations": migrations,
            }
        )
    return output_rows
