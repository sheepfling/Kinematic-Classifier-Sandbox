# BLK-003 — ReV-StED License and Authoritative Validation

## Evidence

Public read access, publisher identity, immutable Git commit, Git blob identity, source header,
and a bounded set of records were inspected. Zenodo labels the record Open, but the inspected
rights information does not identify a concrete license.

A local two-view fixture passed the packet-local structural validator. The authoritative
COMMON-FRONT runtime was not available, so that result is not `fixture_validated`.

## Repository disposition

The public branch excludes ReV-StED source records and derived coordinate arrays. It retains
source identity, mapping rules, aggregate results, and non-reconstructive hashes only.

## Required decisions

1. Establish the governing ReV-StED license or obtain permission.
2. Supply and run the authoritative COMMON-FRONT validator.
3. Decide whether the local-only contract witness may remain accepted evidence until rights
   closure.
