"""Python port of the frontend compatibility engine (computer_shop_ui/src/lib/compatibility.ts).

Operates on a build = {slot: product} where each product carries an `attributes`
dict. Returns the lists of error and warning messages; features.py uses the
counts. Kept in sync with the TS rules so training and the live shop agree.
"""

PSU_HEADROOM = 0.2


def _a(product: dict | None) -> dict:
    """Attributes dict for a product (empty if missing)."""
    if not product:
        return {}
    return product.get("attributes") or {}


def estimated_draw(build: dict[str, dict]) -> int:
    """Rough system draw: CPU + GPU TDP plus a fixed overhead."""
    cpu = _a(build.get("processors")).get("tdp_w") or 0
    gpu = _a(build.get("graphics-cards")).get("tdp_w") or 0
    if cpu == 0 and gpu == 0:
        return 0
    return cpu + gpu + 100


def evaluate(build: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for the selected parts. Rules self-skip when
    the parts or data they need are absent."""
    errors: list[str] = []
    warnings: list[str] = []

    cpu = _a(build.get("processors"))
    mobo = _a(build.get("motherboards"))
    cooler = _a(build.get("cpu-coolers"))
    ram = _a(build.get("memory"))
    gpu = _a(build.get("graphics-cards"))
    psu = _a(build.get("power-supplies"))
    case = _a(build.get("cases"))

    # CPU socket vs motherboard socket
    if cpu.get("socket") and mobo.get("socket") and cpu["socket"] != mobo["socket"]:
        errors.append("cpu_mobo_socket")

    # Cooler supports the CPU socket
    if cooler.get("sockets") and cpu.get("socket") and cpu["socket"] not in cooler["sockets"]:
        errors.append("cooler_socket")

    # Air cooler height vs case
    if (
        cooler.get("cooler_type") == "air"
        and cooler.get("height_mm") is not None
        and case.get("max_cooler_height_mm") is not None
        and cooler["height_mm"] > case["max_cooler_height_mm"]
    ):
        errors.append("cooler_height")

    # AIO radiator vs case
    if (
        cooler.get("cooler_type") == "aio"
        and cooler.get("radiator_mm") is not None
        and case.get("max_radiator_mm") is not None
        and cooler["radiator_mm"] > case["max_radiator_mm"]
    ):
        errors.append("radiator_size")

    # RAM type vs motherboard
    if ram.get("memory_type") and mobo.get("memory_type") and ram["memory_type"] != mobo["memory_type"]:
        errors.append("ram_type")

    # RAM stick count vs motherboard slots
    if ram.get("modules") is not None and mobo.get("memory_slots") is not None and ram["modules"] > mobo["memory_slots"]:
        errors.append("ram_slots")

    # Motherboard form factor vs case
    if mobo.get("form_factor") and case.get("form_factors") and mobo["form_factor"] not in case["form_factors"]:
        errors.append("mobo_form_factor")

    # GPU length vs case
    if gpu.get("length_mm") is not None and case.get("max_gpu_length_mm") is not None and gpu["length_mm"] > case["max_gpu_length_mm"]:
        errors.append("gpu_length")

    # PSU form factor vs case
    if psu.get("form_factor") and case.get("psu_form_factors") and psu["form_factor"] not in case["psu_form_factors"]:
        errors.append("psu_form_factor")

    # PSU wattage vs estimated draw (with headroom warning)
    if psu.get("wattage_w") is not None:
        draw = estimated_draw(build)
        if draw > 0:
            if psu["wattage_w"] < draw:
                errors.append("psu_underpowered")
            elif psu["wattage_w"] < draw * (1 + PSU_HEADROOM):
                warnings.append("psu_headroom")

    # Cooler can handle the CPU heat
    if cooler.get("tdp_rating_w") is not None and cpu.get("tdp_w") is not None and cooler["tdp_rating_w"] < cpu["tdp_w"]:
        warnings.append("cooler_underrated")

    # RAM capacity vs motherboard maximum
    if ram.get("capacity_gb") is not None and mobo.get("memory_max_gb") is not None and ram["capacity_gb"] > mobo["memory_max_gb"]:
        warnings.append("ram_capacity")

    # CPU with no integrated graphics and no discrete GPU
    if build.get("processors") and cpu.get("has_igpu") is False and not build.get("graphics-cards"):
        warnings.append("no_display")

    return errors, warnings
