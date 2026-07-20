"""Plot the benchmark results."""

# %%
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from plotting.hero_charts import Series, grouped_hero_bar
from generate_test_cases import load_citations

if TYPE_CHECKING:
    from mellea_lrc.extraction import ExtractedDocument


# %%
def get_recall(ground_truth: Path, testing: Path) -> tuple[int, int]:
    """Return the results for each model."""
    correct_citations: ExtractedDocument = load_citations(ground_truth)
    model_citations: ExtractedDocument = load_citations(testing)
    correct: int = 0
    total: int = len(correct_citations.full_citations)

    # For now, compare the mateched text
    for ground, model in zip(correct_citations.full_citations, model_citations.full_citations, strict=False):
        if ground.matched_text == model.matched_text:
            correct += 1

    return (total, correct)


# %%
def main() -> int:
    """Retrieve results and generate the plots."""

    working_dir = Path(__file__).parent
    styles_file = working_dir / "plotting" / "deep_hero.mplstyle"
    if not (styles_file.exists() and styles_file.is_file()):
        msg = "Path to the sytles file isn't incorrect"
        raise Exception(msg)
    citation_dir = working_dir / ".cache"
    citation_files = list(filter(lambda x: x.suffix == ".json", citation_dir.iterdir()))

    total, _ = get_recall(citation_files[0], citation_files[0])
    plt.style.use(styles_file)

    result_raw_llm = (11 / 13) * 100
    eyecite_raw = (10 / 13) * 100

    categories = ["MOCK RECALL"]
    sublabels = [f"RECALL@{total}"]

    series = [
        Series("EYECITE-RAW", [eyecite_raw], hero=True),
        Series("LLM-26B-RAW", [result_raw_llm]),
    ]

    fig, ax = plt.subplots()

    grouped_hero_bar(
        ax, categories, series, sublabels=sublabels, ylabel="Recall / Percentile (%)", ylim=(0, 100)
    )

    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.07, right=0.99)
    fig.savefig("recall_1.svg", format="svg")
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())

# %%
