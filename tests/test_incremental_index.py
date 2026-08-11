import unittest
from types import SimpleNamespace

from rag.index import (
    INDEX_SIGNATURE,
    ExistingPoint,
    apply_plan,
    chunk_payload,
    content_hash,
    plan_sync,
    point_id,
    read_existing,
)


def chunk(chunk_id: str, text: str = "same text") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "source_type": "manufacturer_datasheet",
        "chunk_type": "spec",
        "title": "Test material",
        "url": "https://example.test/material",
        "text": text,
        "metadata": {"category": "test"},
    }


def existing(point, value: dict, *, signed: bool = True) -> ExistingPoint:
    payload = chunk_payload(value)
    if signed:
        payload["_content_hash"] = content_hash(payload)
        payload["_index_signature"] = INDEX_SIGNATURE
    return ExistingPoint(point, payload)


class PlanSyncTests(unittest.TestCase):
    def test_classifies_add_update_unchanged_and_stale(self):
        same = chunk("doc:same#spec")
        changed = chunk("doc:changed#spec", "new text")
        old_changed = chunk("doc:changed#spec", "old text")
        stale = chunk("doc:stale#spec")

        plan = plan_sync(
            [same, changed, chunk("doc:added#spec")],
            [
                existing(10, same),
                existing(11, old_changed),
                existing(12, stale),
            ],
        )

        self.assertEqual([10], [point for point, _ in plan.unchanged])
        self.assertEqual([11], [point for point, _ in plan.updated])
        self.assertEqual([12], plan.stale)
        self.assertEqual(point_id("doc:added#spec"), plan.added[0][0])

    def test_legacy_point_is_adopted_without_reembedding(self):
        value = chunk("doc:legacy#spec")
        plan = plan_sync([value], [existing(47, value, signed=False)])
        self.assertEqual([47], [point for point, _ in plan.unchanged])
        self.assertEqual([47], plan.adopted)
        self.assertEqual(0, plan.embeds)

    def test_embedding_signature_change_forces_update(self):
        value = chunk("doc:model#spec")
        record = existing(3, value)
        record.payload["_index_signature"] = "old-model-signature"
        plan = plan_sync([value], [record])
        self.assertEqual([3], [point for point, _ in plan.updated])

    def test_rejects_duplicate_input_ids(self):
        value = chunk("doc:duplicate#spec")
        with self.assertRaisesRegex(ValueError, "duplicate chunk_id"):
            plan_sync([value, value], [])

    def test_rejects_duplicate_collection_ids(self):
        value = chunk("doc:duplicate#spec")
        with self.assertRaisesRegex(RuntimeError, "rebuild with --drop"):
            plan_sync([value], [existing(1, value), existing(2, value)])

    def test_point_id_is_stable_and_uuid_shaped(self):
        first = point_id("smoothon:test#spec")
        self.assertEqual(first, point_id("smoothon:test#spec"))
        self.assertNotEqual(first, point_id("smoothon:test#narrative-0"))
        self.assertEqual(36, len(first))


class FakeClient:
    def __init__(self):
        self.calls = []
        self.pages = []

    def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))

    def set_payload(self, **kwargs):
        self.calls.append(("set_payload", kwargs))

    def scroll(self, **kwargs):
        self.calls.append(("scroll", kwargs))
        return self.pages.pop(0)


class ApplyPlanTests(unittest.TestCase):
    def test_writes_changes_before_pruning_stale_points(self):
        value = chunk("doc:new#spec")
        plan = plan_sync([value], [existing(9, chunk("doc:old#spec"))])
        client = FakeClient()

        apply_plan(client, plan, batch_size=1)

        self.assertEqual(["upsert", "delete"], [name for name, _ in client.calls])
        point = client.calls[0][1]["points"][0]
        self.assertEqual(point_id(value["chunk_id"]), point.id)
        self.assertEqual(INDEX_SIGNATURE, point.payload["_index_signature"])

    def test_no_prune_keeps_stale_points(self):
        plan = plan_sync([], [existing(9, chunk("doc:old#spec"))])
        client = FakeClient()
        apply_plan(client, plan, prune=False)
        self.assertEqual([], client.calls)

    def test_legacy_adoption_persists_signature_without_embedding(self):
        value = chunk("doc:legacy#spec")
        plan = plan_sync([value], [existing(47, value, signed=False)])
        client = FakeClient()
        apply_plan(client, plan)
        self.assertEqual(["set_payload"], [name for name, _ in client.calls])
        call = client.calls[0][1]
        self.assertEqual([47], call["points"])
        self.assertEqual(INDEX_SIGNATURE, call["payload"]["_index_signature"])

    def test_scroll_reads_every_page_without_vectors(self):
        client = FakeClient()
        client.pages = [
            ([SimpleNamespace(id=1, payload={"chunk_id": "one"})], 1),
            ([SimpleNamespace(id=2, payload={"chunk_id": "two"})], None),
        ]
        result = read_existing(client, page_size=1)
        self.assertEqual([1, 2], [record.id for record in result])
        self.assertTrue(all(not call[1]["with_vectors"] for call in client.calls))


if __name__ == "__main__":
    unittest.main()
