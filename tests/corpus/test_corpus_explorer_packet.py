from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kinematic_classifier_sandbox.corpus.validation import validate_corpus_explorer_packet


class CorpusExplorerPacketValidationTests(unittest.TestCase):
    def test_validator_accepts_minimal_valid_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet_dir = Path(temp_dir) / "corpus_explorer_mvp"
            _write_valid_packet(packet_dir)
            self.assertEqual(validate_corpus_explorer_packet(packet_dir), [])

    def test_validator_rejects_selected_candidate_with_leakage_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet_dir = Path(temp_dir) / "corpus_explorer_mvp"
            _write_valid_packet(packet_dir)
            leakage_path = packet_dir / "leakage_adequacy_audit.csv"
            lines = leakage_path.read_text(encoding="utf-8").splitlines()
            lines[1] = lines[1].replace(",pass,low,True,", ",fail,low,True,")
            leakage_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            issues = validate_corpus_explorer_packet(packet_dir)
            self.assertTrue(any("must pass leakage audit" in issue for issue in issues))


def _write_valid_packet(packet_dir: Path) -> None:
    (packet_dir / "figures").mkdir(parents=True, exist_ok=True)
    (packet_dir / "hard_case_cards").mkdir(parents=True, exist_ok=True)

    manifest = {
        "packet_id": "corpus_explorer_mvp",
        "plan_path": "docs/plans/PLN-036_corpus_explorer_execution_brief.md",
        "short_goal_blurb": "Implement V5C Corpus Explorer MVP as a corpus decision system, not a data-generator demo.",
        "validator": (
            "PYTHONPATH=src python3 -m kinematic_classifier_sandbox validate-packet "
            "artifacts/packets/corpus_explorer_mvp --profile corpus_explorer_mvp"
        ),
    }
    _write(packet_dir / "packet_manifest.json", json.dumps(manifest, indent=2) + "\n")
    _write(packet_dir / "README.md", "# V5C\n\ndecision endpoint\n")
    _write(
        packet_dir / "corpus_explorer_decision_card.md",
        "\n".join(
            [
                "# Corpus Explorer Decision Card",
                "status: promote_selected_corpus",
                "cem: experimental_or_run_backed",
                "ppo: experimental_witness",
                "future_3d_lift",
            ]
        )
        + "\n",
    )
    _write(packet_dir / "corpus_objective.yaml", "corpus_objective: {}\n")
    _write(
        packet_dir / "corpus_candidate_frontier.csv",
        "\n".join(
            [
                "candidate_id,generator_backend,source_candidate_id,scenario_family,class_pair_target,validity_status,coverage_score,boundary_stress_score,feature_excitation_score,leakage_status,downstream_yield_score,selected,rejection_reason,target_failure_mode,routed_action",
                "HC001,controlled_1d,src1,switching_case,stationary_vs_slow_velocity,pass,0.9,0.8,0.7,pass,0.75,True,,transition_switching_delay,route to IMM switching witness",
                "RC001,controlled_1d,src2,switching_case,stationary_vs_slow_velocity,fail,0.2,0.7,0.0,blocked,0.0,False,invalid_candidate_rule_check_failed,invalid_generated_hard_case,reject candidate before ladder use",
            ]
        )
        + "\n",
    )
    _write(
        packet_dir / "selected_corpus_manifest.csv",
        "\n".join(
            [
                "candidate_id,generator_backend,source_candidate_id,scenario_family,class_pair_target,validity_status,coverage_score,boundary_stress_score,feature_excitation_score,leakage_status,downstream_yield_score,selected,rejection_reason,target_failure_mode,routed_action",
                "HC001,controlled_1d,src1,switching_case,stationary_vs_slow_velocity,pass,0.9,0.8,0.7,pass,0.75,True,,transition_switching_delay,route to IMM switching witness",
            ]
        )
        + "\n",
    )
    _write(packet_dir / "corpus_adequacy_report.md", "# Adequacy\n")
    _write(
        packet_dir / "leakage_adequacy_audit.csv",
        "\n".join(
            [
                "candidate_id,class_validity,feature_availability,leakage_status,generator_artifact_risk,selected,rejection_reason",
                "HC001,pass,pass,pass,low,True,",
                "RC001,fail,blocked,blocked,high,False,invalid_candidate_rule_check_failed",
            ]
        )
        + "\n",
    )
    _write(
        packet_dir / "feature_excitation_report.csv",
        "\n".join(
            [
                "candidate_id,scenario_family,feature_excitation_score,target_failure_mode,status",
                "HC001,switching_case,0.7,transition_switching_delay,usable",
                "RC001,switching_case,0.0,invalid_generated_hard_case,rejected",
            ]
        )
        + "\n",
    )
    _write(
        packet_dir / "search_backend_comparison.csv",
        "\n".join(
            [
                "backend_id,role,status,valid_discovery_proxy,boundary_stress_proxy,sample_efficiency_proxy,seed_count,diagnostic_yield,promotion_gate,justification",
                "doe_schedule_bank,baseline,baseline,0.7,0.5,0.1,2,baseline_high,baseline_reference,Reference family.",
                "guided_schedule_mutation,baseline,baseline,0.68,0.52,0.11,2,baseline_high,baseline_reference,Reference family.",
                "cem_open_loop,search_backend,experimental,0.73,0.7,0.14,2,medium,not_promoted_without_baseline_ablation_seed_stability_and_downstream_yield,Interpretable search backend.",
                "ppo_policy,search_backend,experimental_witness,0.74,0.71,0.15,2,candidate,not_promoted_without_baseline_ablation_seed_stability_and_downstream_yield,Sequential control witness.",
            ]
        )
        + "\n",
    )
    _write(
        packet_dir / "downstream_diagnostic_yield.csv",
        "\n".join(
            [
                "target_failure_mode,valid_cases,primary_route,decision_trigger",
                "transition_switching_delay,1,route to IMM switching witness,switching_state_failure -> IMM witness",
                "nonlinear_posterior_candidate,1,route to PF/GSF nonlinear-posterior witness,nonlinear_posterior_candidate -> PF/GSF witness",
                "maneuver_vs_oscillatory_confusion,1,route to RBPF latent-event witness after ladder stress run,maneuver ambiguity -> RBPF latent-event witness",
                "invalid_generated_hard_case,0,reject candidate,1 rejected before ladder influence",
            ]
        )
        + "\n",
    )
    _write(packet_dir / "novelty_to_filter_escalation_report.md", "# Escalation\n")
    _write(
        packet_dir / "advanced_algorithm_route_matrix.csv",
        "\n".join(
            [
                "route_id,failure_mode,valid_case_count,advanced_algorithm,method_id,route_status,method_validation_status,trace_status,decision_card_status,supporting_artifact,claim_boundary,why_it_matters_for_3d_lift",
                "switching_state_route,transition_switching_delay,1,IMM,imm_v1,active_route_proof,witness_supported,trace_validated,promote,artifacts/imm_filter_v1/switching_detection_metrics.csv,witness-specific promotion; not a universal default,3D lift",
                "nonlinear_posterior_route,nonlinear_posterior_candidate,1,Particle filter / Gaussian-sum frontier,particle_filter_bank_v1,active_route_proof,study_justified,trace_validated,promote,artifacts/pf/report.csv,witness-specific promotion; not a universal default,3D lift",
                "latent_event_route,maneuver_vs_oscillatory_confusion,1,RBPF,rbpf_v1,active_route_proof,witness_supported,trace_validated,promote,artifacts/rbpf/report.csv,witness-specific promotion; not a universal default,3D lift",
                "stochastic_dynamics_route,stationary_slow_velocity_boundary,1,OU/PF stochastic-dynamics witness,ornstein_uhlenbeck_pf_v1,active_route_proof,witness_supported,trace_validated,promote,artifacts/oupf/report.csv,witness-specific promotion; not a universal default,3D lift",
                "representation_learning_route,handcrafted_feature_underfit,0,TS2Vec-style embedding frontier,ts2vec,available_witness_route,witness_supported,witness_supported,candidate,artifacts/embedding_baseline_frontier_v1/embedding_baseline_frontier_report.md,bounded parity witness exists; not a broad finished claim,3D lift",
            ]
        )
        + "\n",
    )
    _write(packet_dir / "advanced_algorithm_route_proof.md", "not a universal default\n")

    _write(
        packet_dir / "hard_case_cards/HC001_transition_switching_delay.md",
        "# Hard Case\n\nroute to IMM switching witness\n",
    )
    _write(
        packet_dir / "hard_case_cards/RC001_invalid_generated_hard_case.md",
        "# Hard Case\n\nreject candidate before ladder use\n\ninvalid_candidate_rule_check_failed\n",
    )

    for figure_name in (
        "03_corpus_candidate_frontier.png",
        "18_leakage_adequacy_audit.png",
        "21_search_backend_comparison_frontier.png",
        "26_downstream_diagnostic_yield.png",
        "27_novelty_to_filter_escalation_bridge.png",
    ):
        (packet_dir / "figures" / figure_name).write_bytes(b"png")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
