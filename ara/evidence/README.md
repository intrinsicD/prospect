# Evidence Index

## WM-001 protocol 1.3.0 formal attempt

- **Disposition**: rejected for formal acceptance; bounded producer mechanism
  evidence retained
- **Tracked review**: `docs/wm001-v130-formal-results.md`
- **Producer root**:
  `bench/world_model_lifecycle/results/formal/fcb544699d62797999aa40aadcb94ff19af0acac39212f438302aad337a8709b/20260718-v130-8cc5a8b-attempt1`
- **Raw result SHA-256**:
  `df10cdb74c3f9048070140a97aef3a9bbf404fa8cd30212ce8bc82cb72f6dc08`
- **Producer manifest SHA-256**:
  `ada094e0f36b095be83fc13fb88669e49e8bec7bd2e253bb31ca59f48e9a495a`
- **Independent audit**:
  `artifacts/wm001-audits/formal-fcb54469-20260718-v130.json`
  (`9186f95e2c466aa2ba8af8ba8a7d767ec65f85a6393d2723c46a160000d03501`)
- **Semantic review SHA-256**:
  `9835b62b6c57b34ff9e18c7eb444961175dda5ea1c9041c20770ce6a00a50434`
- **Rejected adjudication**:
  `artifacts/wm001-adjudications/formal-fcb54469-20260718-v130-rejected`
  (manifest
  `25c454ddb1764825a932a70b91594ef606f65e6504c4db75f612be8e27a46c1a`)
- **Forensic nodes**: N02, N03, N04, N05
- **Crystallized claim**: C01

## WM-001 protocol 1.4.0 formal attempt

- **Disposition**: retired without formal acceptance; bounded producer and
  direct-audit mechanism evidence retained
- **Tracked review**: `docs/wm001-v140-formal-results.md`
- **Formal binding**:
  `artifacts/wm001-binding-20260718-v140-confirmation/formal-binding.json`
  (`96628f7d551f14f50108a51ee454fe725e7e439501ca5ffe081cf891c0f17857`)
- **Producer root**:
  `bench/world_model_lifecycle/results/formal/96628f7d551f14f50108a51ee454fe725e7e439501ca5ffe081cf891c0f17857/20260718-v140-confirmation`
- **Raw result SHA-256**:
  `bd759ac621494a732ead40b770c66c725cb84e1597085ec674cb747e2891bab0`
- **Producer manifest SHA-256**:
  `af0a7702708b64fe95ecb2e888d39c9262971b921829b8a6f3a680b92008299a`
- **Direct corrected audit**:
  `artifacts/wm001-audits/20260718-v140-formal-confirmation.json`
  (`e1b00f03afaab896db01da5de0991ff32ba46ecdfe909853d93b6a1b0bc9af28`)
- **Accepted pre-adjudication semantic review**:
  `artifacts/wm001-audits/20260718-v140-formal-semantic-review.json`
  (`64427141d825615a183e311ca01dd19608bced63d70383381e02914923e20f87`)
- **Descriptor-bound failing audit replay**:
  `artifacts/wm001-audits/20260719-v140-adjudication-replay-diagnostic-1.json`
  (`11ac22d15db89e43b002ab05223ead05843a4abd55de45fe8777ab702f7dd226`)
- **Superseding rejected semantic review**:
  `artifacts/wm001-audits/20260719-v140-formal-semantic-review-rejected.json`
  (`0ff74453796b2fe447ecf43554152c5ea026215690fa31d39f36da58643112c8`)
- **Adjudication packages**: accepted and rejected intended outputs both absent
- **Forensic nodes**: N07, N08, N09, N10
- **Crystallized claim**: C02

## WM-001 protocol 1.16.0 engineering attempt

- **Disposition**: retired at the mandatory accepted-binding pre-root rehearsal;
  no formal authority or capability result exists
- **Tracked prospective review**:
  `docs/wm001-v1160-prospective-harness-review.json`
- **Tracked terminal review**:
  `docs/wm001-v1160-accepted-binding-rehearsal-failure.md`
- **Sealed commit**:
  `2b5dc659a8a8db872f5d3e6d9655d5da307e857b`
- **Protocol SHA-256**:
  `ac7a8aa331f15412c80a1dad6af9b30c154db33b6d313940e8d2ee546b57dc00`
- **Claim-ineligible development result**: retained opaque at 320,935,092
  bytes; SHA-256
  `90868e3d3e1ca2368758251695569f60650042fd53f5255399b358923d1d82d8`
- **Accepted formal binding SHA-256**:
  `437ad669deaf02c78e261318a2d40d67847c24ab5007c9b0dc9d0e93b16f5104`
- **Rehearsal diagnostic capture**:
  `/tmp/prospect-wm001-v116-binding-rehearsal.na20hR`; stdout was empty and
  stderr was 1,395 bytes with SHA-256
  `e2d081dfb354c5c80da6f9fde4301433c375cc10f863d9f8d466e9a464185884`.
  This temporary capture does not independently authenticate the return code or
  exactly-once execution.
- **Formal evidence**: binding-keyed formal root, marker, producer, outcome,
  audit, review, and adjudication all absent
- **Forensic nodes**: N23, N24, N25, N26
- **Crystallized claim**: C04
