from __future__ import annotations

from dataclasses import replace

from .story import repo_story as _impl
from .utils.runtime import repo_root

ROOT = repo_root()


def _resolve_story_path(path: str) -> str:
    return str((ROOT / path).resolve()) if not path.startswith("/") else path


ARTIFACT_MANIFEST = _impl.ARTIFACT_MANIFEST
CLAIMS = _impl.CLAIMS
LADDER = _impl.LADDER
RepoStoryArtifacts = _impl.RepoStoryArtifacts
WitnessProblem = _impl.WitnessProblem
WITNESSES = tuple(
    replace(
        witness,
        key_plot=_resolve_story_path(witness.key_plot),
        key_table=_resolve_story_path(witness.key_table),
    )
    for witness in _impl.WITNESSES
)
render_proof_gallery = _impl.render_proof_gallery
render_repo_story_index = _impl.render_repo_story_index
render_story_index = _impl.render_story_index
render_team_packet_index = _impl.render_team_packet_index
validate_repo_story_references = _impl.validate_repo_story_references
write_repo_story_artifacts = _impl.write_repo_story_artifacts

__all__ = [
    "ARTIFACT_MANIFEST",
    "CLAIMS",
    "LADDER",
    "RepoStoryArtifacts",
    "WitnessProblem",
    "WITNESSES",
    "render_proof_gallery",
    "render_repo_story_index",
    "render_story_index",
    "render_team_packet_index",
    "validate_repo_story_references",
    "write_repo_story_artifacts",
]
