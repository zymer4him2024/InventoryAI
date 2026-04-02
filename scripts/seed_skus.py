"""One-time script to seed inventory_skus collection in Firestore."""

import sys

import firebase_admin
from firebase_admin import credentials, firestore

SKUS = [
    {
        "sku": "BOLT-M6-10",
        "part_class": "bolt_m6",
        "target_count": 10,
        "required_classes": ["bolt_m6"],
        "customer_id": "default",
        "tolerance": 1,
    },
    {
        "sku": "BOLT-M6-30",
        "part_class": "bolt_m6",
        "target_count": 30,
        "required_classes": ["bolt_m6"],
        "customer_id": "default",
        "tolerance": 2,
    },
    {
        "sku": "KIT-A",
        "part_class": "",
        "target_count": 0,
        "required_classes": ["bolt_m6", "washer_m6", "nut_m6"],
        "customer_id": "default",
        "tolerance": 0,
    },
    {
        "sku": "WASHER-M6-20",
        "part_class": "washer_m6",
        "target_count": 20,
        "required_classes": ["washer_m6"],
        "customer_id": "default",
        "tolerance": 1,
    },
]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/seed_skus.py <path-to-firebase-credentials.json>")
        sys.exit(1)

    cred_path = sys.argv[1]
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    for sku_data in SKUS:
        doc_id = sku_data["sku"]
        db.collection("inventory_skus").document(doc_id).set(sku_data)
        print(f"  Seeded: {doc_id}")

    print(f"\nDone. {len(SKUS)} SKUs written to inventory_skus.")


if __name__ == "__main__":
    main()
