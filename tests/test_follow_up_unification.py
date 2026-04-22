import unittest
from unittest.mock import patch

from engine import agents
from engine import orchestration_service as orchestration


class FollowUpUnificationTests(unittest.TestCase):
    def test_follow_up_router_dispatches_transfer(self) -> None:
        with patch.object(orchestration, "orchestrate_transfer_chat_turn", return_value={"status": "ok", "final_report": "t"}) as mocked_transfer:
            result = orchestration.orchestrate_follow_up_chat_turn(
                user_prompt="latest updates?",
                current_state={"transfer_report_context": {"player_name": "Sample"}},
                portal="transfer",
                allow_web_refresh=False,
            )

        self.assertEqual("ok", result.get("status"))
        mocked_transfer.assert_called_once()
        self.assertFalse(mocked_transfer.call_args.kwargs["allow_web_refresh"])

    def test_follow_up_router_sets_allow_web_refresh_for_recruiting(self) -> None:
        with patch.object(orchestration, "orchestrate_chat_turn", return_value={"final_report": "r"}) as mocked_recruiting:
            result = orchestration.orchestrate_follow_up_chat_turn(
                user_prompt="summarize this player",
                current_state={"target_player_name": "Sample"},
                portal="recruiting",
                allow_web_refresh=False,
            )

        self.assertEqual("ok", result.get("status"))
        mocked_recruiting.assert_called_once()
        routed_state = mocked_recruiting.call_args.kwargs["current_state"]
        self.assertFalse(routed_state.get("allow_web_refresh", True))

    def test_recruiting_delegator_respects_explicit_disable(self) -> None:
        mock_plan = {
            "cfbd_search_params": {},
            "recruiting_web_query": "sample player transfer news",
            "team_context_query": "sample team depth chart",
            "user_intent": "Need recency",
        }
        state = {
            "mode": "chat",
            "user_query": "latest updates",
            "allow_web_refresh": False,
            "trace_log": [],
        }

        with patch.object(agents, "delegator_plan_tool", return_value=mock_plan), patch.object(agents, "_infer_chat_route", return_value="web_scout"):
            updated = agents.lead_delegator_node(state)

        self.assertFalse(updated.get("allow_web_refresh", True))
        plan = dict(updated.get("delegator_plan") or {})
        self.assertEqual("", plan.get("recruiting_web_query"))
        self.assertEqual("", plan.get("team_context_query"))

    def test_parallel_web_scout_skips_when_delegator_disables_refresh(self) -> None:
        state = {
            "mode": "chat",
            "allow_web_refresh": False,
            "requires_identity_clarification": False,
            "trace_log": [],
        }

        updates = agents.parallel_web_scout_node(state)

        self.assertFalse(updates.get("web_recruiting_used", True))
        self.assertFalse(updates.get("web_team_used", True))
        trace_log = list(updates.get("trace_log") or [])
        trace_text = "\n".join(str(item) for item in trace_log)
        self.assertIn("skipped_delegator_no_web_refresh", trace_text)


if __name__ == "__main__":
    unittest.main()
