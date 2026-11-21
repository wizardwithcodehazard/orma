"""
test_loci.py - Tests for the Method of Loci (Topological Memory Palace) module.
"""

import tempfile
from pathlib import Path

from orma import Memory, OrmaConfig
from orma.loci import PalaceWalkResult


def test_loci_creation_and_pegging():
    with tempfile.TemporaryDirectory() as td:
        mem = Memory(config=OrmaConfig(data_dir=Path(td)))
        try:
            # 1. Establish Memory Palace Checkpoints (Rooms)
            loc1 = mem.checkpoint(label="Session 1", gist="Job loss and emotional struggle")
            assert loc1.ordinal == 1
            assert loc1.label == "Session 1"

            # Observe turns pegged to Locus 1
            mem.observe("user", "I lost my job as an architect yesterday.")
            mem.observe("assistant", "I am so sorry to hear that.")

            # 2. Establish next Palace Checkpoint
            loc2 = mem.checkpoint(label="Session 2", gist="Career exploration and baking")
            assert loc2.ordinal == 2

            mem.observe("user", "I am thinking about opening an artisan bakery.")
            mem.observe("assistant", "That sounds like a wonderful path forward.")

            # 3. Establish Locus 3
            loc3 = mem.checkpoint(label="Session 3", gist="Bakery launch with partner Alex")
            assert loc3.ordinal == 3

            mem.observe("user", "Alex agreed to partner with me on the bakery.")

            # List all palace checkpoints
            loci = mem.palace.list_loci("default")
            assert len(loci) == 3
            assert [loc.ordinal for loc in loci] == [1, 2, 3]

            # 4. Perform Palace Walk on Multi-Hop query
            # Query targeting Session 2 ("baking"), with radius=1 walks Session 1, 2, 3!
            walk_res = mem.walk_palace("baking", radius=1)
            assert isinstance(walk_res, PalaceWalkResult)
            assert len(walk_res.walked_loci) == 3
            assert len(walk_res.sentences) >= 3
            assert "Session 1" in walk_res.narrative_text
            assert "Session 2" in walk_res.narrative_text
            assert "Session 3" in walk_res.narrative_text
        finally:
            mem.close()


def test_loci_empty_palace():
    with tempfile.TemporaryDirectory() as td:
        mem = Memory(config=OrmaConfig(data_dir=Path(td)))
        try:
            assert mem.palace.current_locus("default") is None
            walk_res = mem.walk_palace("random query")
            assert isinstance(walk_res, PalaceWalkResult)
            assert len(walk_res.walked_loci) == 0
            assert len(walk_res.states) == 0
            assert len(walk_res.sentences) == 0
        finally:
            mem.close()


def test_loci_state_pegging_and_fallback():
    with tempfile.TemporaryDirectory() as td:
        mem = Memory(config=OrmaConfig(data_dir=Path(td)))
        try:
            l1 = mem.checkpoint("Room A", gist="Quantum physics intro")
            sub_id = mem.store.add_entity("default", "User", "person")
            attr_id = mem.store.add_attribute("default", "study")
            res = mem.observe("user", "I studied quantum tunneling in physics.")
            assert res.episode_id is not None
            sents = mem.store.episode_sentences([res.episode_id])
            sent_id = sents[0].id if sents else 1
            st_id = mem.store.add_state(
                "default", sub_id, attr_id, "studied quantum tunneling", 100.0, sent_id, 100.0
            )
            mem.palace.peg_state(l1.id, st_id)

            l2 = mem.checkpoint("Room B", gist="Relativity and gravitation")
            st_id2 = mem.store.add_state(
                "default", sub_id, attr_id, "Einstein field equations", 200.0, sent_id, 200.0
            )
            mem.palace.peg_state(l2.id, st_id2)

            # Checkpoint lookup
            cur = mem.palace.current_locus("default")
            assert cur is not None
            assert cur.ordinal == 2
            assert cur.label == "Room B"

            # Query matching Room A specifically
            walk_res = mem.walk_palace("quantum", radius=0)
            assert len(walk_res.walked_loci) == 1
            assert walk_res.walked_loci[0].label == "Room A"
            assert len(walk_res.states) == 1
            assert "studied quantum tunneling" in walk_res.narrative_text

            # Query semantic matching without keyword overlap (pure vector similarity)
            walk_semantic = mem.walk_palace("subatomic particle mechanics", radius=0)
            assert len(walk_semantic.walked_loci) == 1
            assert walk_semantic.walked_loci[0].label == "Room A"
            assert "studied quantum tunneling" in walk_semantic.narrative_text

            # Query fallback (unrelated concept) -> topological walk handles graceful resolution
            walk_fallback = mem.walk_palace("xyz123 unmatched", radius=0)
            assert len(walk_fallback.walked_loci) == 1
            assert walk_fallback.walked_loci[0].label == "Room B"
            assert "Einstein field equations" in walk_fallback.narrative_text
        finally:
            mem.close()


def test_loci_multilingual_semantic_walk():
    with tempfile.TemporaryDirectory() as td:
        mem = Memory(config=OrmaConfig(data_dir=Path(td)))
        try:
            # Checkpoints with multilingual descriptions
            mem.checkpoint("Room Baking", gist="Ghari navin bakery business aani paav banavne")
            mem.observe("user", "Aamhi bakery chalu keli ahe.")

            mem.checkpoint(
                "Room Education", gist="College shikshan, engineering degree aani abhyas"
            )
            mem.observe("user", "Engineering college madhe admission ghetle.")

            # Cross-lingual query: English query targeting Marathi bakery gist
            walk_baking = mem.walk_palace("artisan bread bakery startup", radius=0)
            assert len(walk_baking.walked_loci) == 1
            assert walk_baking.walked_loci[0].label == "Room Baking"

            # Marathi query targeting education checkpoint
            walk_edu = mem.walk_palace("college shikshan abhyas", radius=0)
            assert len(walk_edu.walked_loci) == 1
            assert walk_edu.walked_loci[0].label == "Room Education"
        finally:
            mem.close()
