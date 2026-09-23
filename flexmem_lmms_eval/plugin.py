"""Entry-point metadata used by lmms-eval to discover FlexMem."""

from lmms_eval.models.registry_v2 import ModelManifest


def get_model_manifests() -> tuple[ModelManifest, ModelManifest]:
    """Expose FlexMem without modifying lmms-eval's installed source tree."""
    return (
        ModelManifest(
            model_id="flexmem",
            simple_class_path="flexmem_lmms_eval.model.FlexMem",
            aliases=("flexmem_stream",),
        ),
        ModelManifest(
            model_id="flexmem_fast",
            simple_class_path="flexmem_lmms_eval.fast_model.FlexMemFast",
            aliases=("flexmem-fast",),
        ),
    )
