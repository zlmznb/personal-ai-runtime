---
title: Local First Architecture
tags: [architecture, privacy]
---

# Local First Architecture

Local-first software keeps the authoritative copy of the user's data on the
user's own device, and treats the network as an optional synchronisation
channel rather than a requirement.

The practical consequences are worth spelling out:

- The application must remain fully usable with no connectivity at all.
- Conflict resolution has to be explicit, because there is no central arbiter
  deciding what "the current version" means.
- Storage format stability matters more than raw storage performance.

## Relation to memory

An AI memory layer that keeps its data only in a vendor's cloud inherits the
vendor's lifetime. Keeping the durable copy in a local file means the layer
outlives any particular model or service.
