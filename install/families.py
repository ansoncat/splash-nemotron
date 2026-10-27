"""The model families Splash serves, the DFlash2 draft released for each and
the chat format the server reads and writes each one's output in.

A target's family is the one the engine finds its configuration to describe,
by the rules it holds every start to (upstream.check_model), never its
repository's name.
"""

from __future__ import annotations

from dataclasses import dataclass

from server.chat_format import FLINT, NEMOTRON, QWEN, ChatFormat


@dataclass(frozen=True)
class ModelFamily:
    # The family's name, as the engine names it (each target layout's family,
    # runtime/model/Qwen3_8.hpp, Qwen3_6Moe.hpp, Qwen4Exp.hpp, Flint.hpp and NemotronH.hpp).
    name: str
    # The repository of the DFlash2 draft released for the family, as its
    # release publishes it (config.json and BF16 safetensors), or as one
    # LiLiCorr GGUF, which installations convert (dflash_gguf.py).
    # Installations follow its default branch as they follow the target's.
    # None while no draft is released for the family: an installation names
    # one with --draft-model.
    draft_repo: str | None
    # How the family's model reasons and calls tools, and the tokens it
    # shares with the engine (server/chat_format.py).
    chat_format: ChatFormat
    # Whether the family's models have a vision tower, which an installation
    # loads from the target repository unless --language-only. The models of
    # a family without one serve text only, whatever their repository holds.
    vision: bool = True
    # The --state-format values the family's recurrent state is kept in
    # (server/serve_options.py); the server refuses another before it starts
    # the engine.
    state_formats: tuple[str, ...] = ("fp32", "bf16")


FAMILIES = (
    ModelFamily("Qwen3.8-27B", "incoai/Qwen3.8-27B-DFlash2", QWEN),
    ModelFamily("Qwen3.6-35B-A3B", "incoai/Qwen3.6-35B-A3B-DFlash2", QWEN),
    # Its fine-tune Ornith-1.5-122B is the family too, and the draft
    # released for it the family's.
    ModelFamily("Qwen3.5-122B-A10B", "incoai/Ornith-1.5-122B-DFlash2", QWEN),
    ModelFamily("Qwen3.8-Flash-Next", "incoai/Qwen3.8-Flash-Next-DFlash2", QWEN),
    # It has no vision tower.
    ModelFamily("Flint", "incoai/Flint-DFlash2", FLINT, vision=False),
    # Its LiLiCorr draft is released as a GGUF only. Its
    # RADIO tower loads from a GGUF projector only. Its Mamba-2 state is
    # fp32 only (runtime/model/NemotronHTarget.cpp).
    ModelFamily(
        "Nemotron-3.5-Bolt",
        "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF",
        NEMOTRON,
        state_formats=("fp32",),
    ),
)


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)
