"""Feature engineering for the FDS Pokemon Battle Prediction challenge.

Every battle is turned into one flat row of numbers. Only the competition
data and the supplied Gen 2 type chart are used.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
STATUSES = ["slp", "par", "tox", "psn", "frz", "brn", "fnt"]
BOOSTS = ["atk", "def", "spa", "spd", "spe"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_type_chart(path):
    """Return dict {(attacking_type, defending_type): multiplier}."""
    df = pd.read_csv(path)
    return {(a, d): m for a, d, m in zip(df.attacking_type, df.defending_type, df.multiplier)}


def type_mult(chart, atk_type, def_types):
    """Damage multiplier of one attacking type against a (possibly dual) defender."""
    m = 1.0
    for d in def_types:
        m *= chart.get((atk_type, d), 1.0)
    return m


def best_mult(chart, atk_types, def_types):
    """Best STAB-type multiplier a Pokemon with atk_types has against def_types."""
    return max(type_mult(chart, a, def_types) for a in atk_types)


def _hp_curve(timeline, side):
    return np.array([t[f"{side}_pokemon_state"]["hp_pct"] for t in timeline], dtype=float)


def _team_state(timeline, side, team_names=None):
    """Last known hp_pct of every Pokemon seen on one side.

    Unseen Pokemon are assumed to be at full HP. Returns (dict name->hp, set of fainted).
    """
    last = {}
    fainted = set()
    for t in timeline:
        s = t[f"{side}_pokemon_state"]
        last[s["name"]] = s["hp_pct"]
        if s["status"] == "fnt" or s["hp_pct"] == 0:
            fainted.add(s["name"])
    return last, fainted


def battle_features(b, chart):
    f = {}
    team = b["p1_team_details"]
    lead2 = b["p2_lead_details"]
    tl = b["battle_timeline"]

    # ---------- static: Player 1 team ----------
    for s in STATS:
        v = np.array([p[f"base_{s}"] for p in team], dtype=float)
        f[f"p1_team_{s}_mean"] = v.mean()
        f[f"p1_team_{s}_max"] = v.max()
        f[f"p1_team_{s}_min"] = v.min()
    bst1 = np.array([sum(p[f"base_{s}"] for s in STATS) for p in team], dtype=float)
    f["p1_team_bst_mean"] = bst1.mean()
    f["p1_team_bst_max"] = bst1.max()
    f["p1_team_bst_min"] = bst1.min()
    f["p1_team_n_fast"] = sum(p["base_spe"] >= 100 for p in team)
    f["p1_team_items_leftovers"] = sum(p["item"] == "leftovers" for p in team)
    f["p1_team_items_other"] = sum(p["item"] != "leftovers" for p in team)

    # ---------- static: Player 2 lead ----------
    for s in STATS:
        f[f"p2_lead_{s}"] = lead2[f"base_{s}"]
    bst2 = sum(lead2[f"base_{s}"] for s in STATS)
    f["p2_lead_bst"] = bst2
    f["bst_team1_mean_minus_lead2"] = bst1.mean() - bst2

    # Player 1 actual lead = Pokemon active at turn 1
    lead1_name = tl[0]["p1_pokemon_state"]["name"]
    lead1 = next((p for p in team if p["name"] == lead1_name), team[0])
    for s in STATS:
        f[f"lead_diff_{s}"] = lead1[f"base_{s}"] - lead2[f"base_{s}"]
    f["lead_speed_adv"] = float(lead1["base_spe"] > lead2["base_spe"])

    # ---------- static: type matchups against the P2 lead ----------
    my_off = [best_mult(chart, p["types"], lead2["types"]) for p in team]
    their_off = [best_mult(chart, lead2["types"], p["types"]) for p in team]
    f["p1_off_vs_lead2_mean"] = np.mean(my_off)
    f["p1_off_vs_lead2_max"] = np.max(my_off)
    f["p1_n_super_vs_lead2"] = sum(m >= 2 for m in my_off)
    f["lead2_off_vs_p1_mean"] = np.mean(their_off)
    f["lead2_off_vs_p1_max"] = np.max(their_off)
    f["p1_n_weak_to_lead2"] = sum(m >= 2 for m in their_off)
    f["p1_n_resist_lead2"] = sum(m < 1 for m in their_off)
    f["lead_type_off_diff"] = best_mult(chart, lead1["types"], lead2["types"]) - best_mult(chart, lead2["types"], lead1["types"])

    # ---------- dynamic: HP curves ----------
    h1, h2 = _hp_curve(tl, "p1"), _hp_curve(tl, "p2")
    f["hp1_last"], f["hp2_last"] = h1[-1], h2[-1]
    f["hp_last_diff"] = h1[-1] - h2[-1]
    for k in (5, 10, 20):
        f[f"hp1_mean_last{k}"] = h1[-k:].mean()
        f[f"hp2_mean_last{k}"] = h2[-k:].mean()
        f[f"hp_diff_mean_last{k}"] = h1[-k:].mean() - h2[-k:].mean()
    f["hp_diff_mean_all"] = h1.mean() - h2.mean()
    f["hp1_min"], f["hp2_min"] = h1.min(), h2.min()

    # damage taken / healed while the same Pokemon stays in
    for side, h in (("p1", h1), ("p2", h2)):
        names = [t[f"{side}_pokemon_state"]["name"] for t in tl]
        dmg = heal = 0.0
        for i in range(1, len(tl)):
            if names[i] == names[i - 1]:
                d = h[i] - h[i - 1]
                if d < 0:
                    dmg += -d
                else:
                    heal += d
        f[f"{side}_dmg_taken"] = dmg
        f[f"{side}_healed"] = heal
        f[f"{side}_n_switches"] = sum(names[i] != names[i - 1] for i in range(1, len(tl)))
        f[f"{side}_n_distinct_seen"] = len(set(names))
    f["dmg_taken_diff"] = f["p1_dmg_taken"] - f["p2_dmg_taken"]
    f["switch_diff"] = f["p1_n_switches"] - f["p2_n_switches"]

    # ---------- dynamic: whole-team state ----------
    last1, fnt1 = _team_state(tl, "p1")
    last2, fnt2 = _team_state(tl, "p2")
    f["p1_n_fainted"], f["p2_n_fainted"] = len(fnt1), len(fnt2)
    f["fainted_diff"] = len(fnt2) - len(fnt1)
    rem1 = sum(last1.values()) + max(0, 6 - len(last1))
    rem2 = sum(last2.values()) + max(0, 6 - len(last2))
    f["p1_team_hp_left"], f["p2_team_hp_left"] = rem1, rem2
    f["team_hp_left_diff"] = rem1 - rem2
    first_f1 = next((i for i, t in enumerate(tl) if t["p1_pokemon_state"]["status"] == "fnt"), 31)
    first_f2 = next((i for i, t in enumerate(tl) if t["p2_pokemon_state"]["status"] == "fnt"), 31)
    f["p1_first_faint_turn"], f["p2_first_faint_turn"] = first_f1, first_f2

    # ---------- dynamic: status, effects, boosts ----------
    for side in ("p1", "p2"):
        sts = [t[f"{side}_pokemon_state"]["status"] for t in tl]
        c = Counter(sts)
        for s in STATUSES:
            f[f"{side}_turns_{s}"] = c.get(s, 0)
        f[f"{side}_last_status_bad"] = float(sts[-1] not in ("nostatus",))
        eff = Counter(e for t in tl for e in t[f"{side}_pokemon_state"]["effects"] if e != "noeffect")
        for e in ("trapped", "confusion", "substitute", "leechseed", "protect", "perish", "attract", "encore", "destinybond", "nightmare"):
            f[f"{side}_eff_{e}"] = eff.get(e, 0)
        bsum_last = sum(tl[-1][f"{side}_pokemon_state"]["boosts"][k] for k in BOOSTS)
        bsum_mean = np.mean([sum(t[f"{side}_pokemon_state"]["boosts"][k] for k in BOOSTS) for t in tl])
        f[f"{side}_boost_last"], f[f"{side}_boost_mean"] = bsum_last, bsum_mean
        for k in BOOSTS:
            f[f"{side}_boost_{k}_last"] = tl[-1][f"{side}_pokemon_state"]["boosts"][k]
    f["status_turns_diff"] = sum(f[f"p2_turns_{s}"] for s in STATUSES[:-1]) - sum(f[f"p1_turns_{s}"] for s in STATUSES[:-1])
    f["boost_mean_diff"] = f["p1_boost_mean"] - f["p2_boost_mean"]

    # ---------- dynamic: moves ----------
    for side in ("p1", "p2"):
        mv = [t[f"{side}_move_details"] for t in tl]
        used = [m for m in mv if m is not None]
        f[f"{side}_n_moves"] = len(used)
        f[f"{side}_n_none"] = len(mv) - len(used)
        cat = Counter(m["category"] for m in used)
        for k in ("physical", "special", "status"):
            f[f"{side}_mv_{k}"] = cat.get(k, 0)
        att = [m["base_power"] for m in used if m["category"] != "status"]
        f[f"{side}_mv_bp_mean"] = np.mean(att) if att else 0.0
        f[f"{side}_mv_prio"] = sum(m["priority"] != 0 for m in used)
        names = Counter(m["name"] for m in used)
        for k in ("rest", "sleeptalk", "curse", "spikes", "toxic", "explosion", "selfdestruct", "recover", "thunder", "earthquake", "surf", "protect", "roar", "whirlwind", "haze", "reflect", "lightscreen", "substitute", "healbell", "perishsong", "rapidspin", "hiddenpower", "doubleedge", "thunderwave"):
            f[f"{side}_use_{k}"] = names.get(k, 0)
        f[f"{side}_n_unique_moves"] = len(names)
    f["moves_used_diff"] = f["p1_n_moves"] - f["p2_n_moves"]

    # ---------- dynamic: who had the type advantage on the field ----------
    adv = []
    for t in tl:
        n1, n2 = t["p1_pokemon_state"]["name"], t["p2_pokemon_state"]["name"]
        p1 = next((p for p in team if p["name"] == n1), None)
        if p1 is None:
            continue
        # the opposing types are unknown unless it is the lead, so use the P2 lead only
        if n2 == lead2["name"]:
            adv.append(best_mult(chart, p1["types"], lead2["types"]) - best_mult(chart, lead2["types"], p1["types"]))
    f["field_type_adv_mean"] = np.mean(adv) if adv else 0.0
    return f


def pokemon_bag(b):
    """Bag-of-Pokemon counts: P1 team members and P2 Pokemon seen in the timeline."""
    d = {}
    for p in b["p1_team_details"]:
        d[f"t1_{p['name']}"] = 1
    seen2 = {t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]}
    seen2.add(b["p2_lead_details"]["name"])
    for n in seen2:
        d[f"s2_{n}"] = 1
    return d


def build_lookup(battles):
    """name -> {types, base stats}, built only from the supplied battle files.

    Player 2 only gets its lead described, but the same Pokemon appear on Player 1
    teams, so we can recover the base stats of Player 2's other Pokemon by name.
    """
    lk = {}
    for b in battles:
        for p in list(b["p1_team_details"]) + [b["p2_lead_details"]]:
            if p["name"] not in lk:
                lk[p["name"]] = {"types": p["types"], **{s: p[f"base_{s}"] for s in STATS}}
    return lk


def extra_features(b, chart, lk):
    """Features that need the name lookup (stats and types of every Pokemon P2 shows)."""
    f = {}
    tl = b["battle_timeline"]
    team = b["p1_team_details"]
    team_by_name = {p["name"]: p for p in team}

    seen2 = []
    for t in tl:
        n = t["p2_pokemon_state"]["name"]
        if n not in seen2:
            seen2.append(n)
    info2 = [lk[n] for n in seen2 if n in lk]
    if info2:
        for s in STATS:
            v = np.array([i[s] for i in info2], dtype=float)
            f[f"p2_seen_{s}_mean"] = v.mean()
        f["p2_seen_bst_mean"] = np.mean([sum(i[s] for s in STATS) for i in info2])
        f["p2_seen_spe_max"] = max(i["spe"] for i in info2)
    else:
        for s in STATS:
            f[f"p2_seen_{s}_mean"] = 0.0
        f["p2_seen_bst_mean"] = 0.0
        f["p2_seen_spe_max"] = 0.0
    f["bst_diff_team_vs_seen"] = np.mean([sum(p[f"base_{s}"] for s in STATS) for p in team]) - f["p2_seen_bst_mean"]

    # per-turn: move effectiveness against the opposing active Pokemon, and stat gaps on the field
    e1, e2, spd_adv, bst_gap = [], [], [], []
    for t in tl:
        n1, n2 = t["p1_pokemon_state"]["name"], t["p2_pokemon_state"]["name"]
        a1, a2 = lk.get(n1), lk.get(n2)
        m1, m2 = t["p1_move_details"], t["p2_move_details"]
        if a1 and a2:
            spd_adv.append(float(a1["spe"] > a2["spe"]) - float(a1["spe"] < a2["spe"]))
            bst_gap.append(sum(a1[s] for s in STATS) - sum(a2[s] for s in STATS))
            if m1 is not None and m1["category"] != "status":
                e1.append(type_mult(chart, m1["type"], a2["types"]))
            if m2 is not None and m2["category"] != "status":
                e2.append(type_mult(chart, m2["type"], a1["types"]))
    f["p1_move_eff_mean"] = np.mean(e1) if e1 else 1.0
    f["p2_move_eff_mean"] = np.mean(e2) if e2 else 1.0
    f["move_eff_diff"] = f["p1_move_eff_mean"] - f["p2_move_eff_mean"]
    f["p1_n_super"] = sum(x >= 2 for x in e1)
    f["p2_n_super"] = sum(x >= 2 for x in e2)
    f["p1_n_resisted"] = sum(x < 1 for x in e1)
    f["p2_n_resisted"] = sum(x < 1 for x in e2)
    f["super_diff"] = f["p1_n_super"] - f["p2_n_super"]
    f["field_speed_adv_mean"] = np.mean(spd_adv) if spd_adv else 0.0
    f["field_bst_gap_mean"] = np.mean(bst_gap) if bst_gap else 0.0
    f["field_bst_gap_last"] = bst_gap[-1] if bst_gap else 0.0

    # momentum: HP change over the last turns
    h1, h2 = _hp_curve(tl, "p1"), _hp_curve(tl, "p2")
    f["hp1_trend3"], f["hp2_trend3"] = h1[-1] - h1[-4], h2[-1] - h2[-4]
    f["hp_diff_trend5"] = (h1[-1] - h2[-1]) - (h1[-6] - h2[-6])

    # active Pokemon on the last turn: stats of both
    a1, a2 = lk.get(tl[-1]["p1_pokemon_state"]["name"]), lk.get(tl[-1]["p2_pokemon_state"]["name"])
    for s in STATS:
        f[f"last_p1_{s}"] = a1[s] if a1 else 0
        f[f"last_p2_{s}"] = a2[s] if a2 else 0
    f["last_status_p1"] = ["nostatus", "slp", "par", "tox", "psn", "frz", "brn", "fnt"].index(tl[-1]["p1_pokemon_state"]["status"])
    f["last_status_p2"] = ["nostatus", "slp", "par", "tox", "psn", "frz", "brn", "fnt"].index(tl[-1]["p2_pokemon_state"]["status"])
    return f


def move_bag(b, top_moves):
    d = {}
    for side in ("p1", "p2"):
        c = Counter(t[f"{side}_move_details"]["name"] for t in b["battle_timeline"] if t[f"{side}_move_details"] is not None)
        for m in top_moves:
            d[f"m{side}_{m}"] = c.get(m, 0)
    return d


def build_frame(battles, chart, lk=None, top_moves=None, with_bag=True):
    rows, bags = [], []
    for b in battles:
        r = battle_features(b, chart)
        if lk is not None:
            r.update(extra_features(b, chart, lk))
        if top_moves is not None:
            r.update(move_bag(b, top_moves))
        rows.append(r)
        if with_bag:
            bags.append(pokemon_bag(b))
    X = pd.DataFrame(rows)
    if with_bag:
        B = pd.DataFrame(bags).fillna(0).astype(np.int8)
        X = pd.concat([X, B], axis=1)
    X.insert(0, "battle_id", [b["battle_id"] for b in battles])
    return X


def top_moves_from(battles, n=60):
    c = Counter(t[f"{s}_move_details"]["name"] for b in battles for t in b["battle_timeline"] for s in ("p1", "p2") if t[f"{s}_move_details"] is not None)
    return [m for m, _ in c.most_common(n)]
