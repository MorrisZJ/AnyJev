"""Games as decision benchmarks with a built-in oracle.

Each game turns one model into a player through one typed question, then plays the same
seeds under the raw next-token readout and under AnyJev (L0, and L1 when the game can label
its own calibration set). The game engine scores every decision against an exact or
near-exact oracle, so the demos measure decision quality, not just final score.

    python -m demo.games.twenty48    --model Qwen/Qwen3-8B --games 3
    python -m demo.games.minesweeper --model Qwen/Qwen3-8B --games 5 --levels raw,L0,L1

`--backend fake` runs either demo on a synthetic biased model, no GPU needed.
"""
