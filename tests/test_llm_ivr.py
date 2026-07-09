"""Tests for project-owned Mellea IVR helpers."""

from mellea.stdlib.requirements import check, req

from mellea_lrc.llm import (
    InstructIvrSpec,
    format_rendered_chat_messages,
    render_instruct_chat_messages,
    render_instruct_prompt,
    visualize_instruct_chat_messages,
)


def test_instruct_ivr_rendering_exposes_raw_prompt_and_chat_messages() -> None:
    spec = InstructIvrSpec(
        description="Extract the answer from local_context.",
        grounding_context={"local_context": "Answer: Smith"},
        requirements=[
            req('Return {"answer":"..."}.'),
            check("answer must be copied from local_context"),
        ],
    )

    prompt = render_instruct_prompt(spec)
    messages = render_instruct_chat_messages(spec)

    assert [message.role for message in messages] == ["user"]
    assert messages[0].content == prompt
    assert 'Return {"answer":"..."}.' in prompt
    assert "answer must be copied" not in prompt
    assert "[local_context]" in prompt
    assert visualize_instruct_chat_messages(spec) == format_rendered_chat_messages(messages)
