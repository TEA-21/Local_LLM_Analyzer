"""CLI application for companion-packager.

Hardware-aware model analyzer, researcher, citation generator, and installer packager
for Agent Companion.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from companion_packager.hardware.detector import check_disk_space_preflight, detect_hardware
from companion_packager.installer.configurator import (
    create_desktop_shortcut,
    inject_agent_companion_config,
    resolve_companion_path,
)
from companion_packager.installer.downloader import (
    DiskSpaceError,
    DownloadChecksumError,
    DownloadError,
    clean_hash,
    compute_sha256,
    download_file,
)
from companion_packager.installer.inno_builder import build_windows_installer
from companion_packager.installer.sfx_builder import build_posix_sfx
from companion_packager.installer.wheel_vendor import (
    acquire_python_standalone,
    build_packager_wheel,
    extract_and_filter_requirements,
    stage_and_build_monorepo_wheels,
    vendor_third_party_wheels,
)
from companion_packager.interview.questionnaire import ask_user_needs
from companion_packager.research.citations import (
    generate_citation_card,
    generate_recommendations_markdown,
)
from companion_packager.research.hf_client import get_model_candidates
from companion_packager.schemas import CandidateModel, HardwareProfile, Scorecard, UserNeeds
from companion_packager.sizing.recommender import display_scorecard_table, rank_candidates

app = typer.Typer(
    name="companion-packager",
    help="Hardware-aware model analyzer, researcher, and packager for Agent Companion.",
    add_completion=False,
)


def version_callback(value: bool) -> None:
    """Print version and exit eagerly."""
    if value:
        typer.echo("companion-packager v0.1.0")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show companion-packager version and exit.",
    ),
) -> None:
    """Companion Packager root options."""
    pass


@app.command("version")
def version_cmd() -> None:
    """Print companion-packager version."""
    typer.echo("companion-packager v0.1.0")


@app.command("doctor")
def doctor() -> None:
    """Validate environment: Python version, SIMD flags, RAM vs 6144 MB ceiling, GPUs, and workspace."""
    console = Console()
    table = Table(
        title="[bold green]Companion Packager System Doctor[/bold green]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Component", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    # 1. Python version check
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11):
        table.add_row("Python Version", "[green]OK[/green]", f"Python {py_ver} (>= 3.11 required)")
    else:
        table.add_row("Python Version", "[red]FAIL[/red]", f"Python {py_ver} (< 3.11 unsupported)")

    # 2. Hardware probing
    hw = detect_hardware()

    # 3. CPU SIMD acceleration
    simd_flags = []
    if hw.has_avx512:
        simd_flags.append("AVX-512")
    if hw.has_avx2:
        simd_flags.append("AVX2")
    if hw.has_neon:
        simd_flags.append("NEON")
    simd_str = ", ".join(simd_flags) if simd_flags else "None"

    if hw.has_avx512 or hw.has_neon:
        table.add_row("CPU SIMD Acceleration", "[green]OPTIMAL[/green]", f"Flags: {simd_str}")
    elif hw.has_avx2:
        table.add_row("CPU SIMD Acceleration", "[green]SUPPORTED[/green]", f"Flags: {simd_str}")
    else:
        table.add_row("CPU SIMD Acceleration", "[yellow]FALLBACK[/yellow]", "No AVX2/AVX-512/NEON detected")

    # 4. RAM vs 6144 MB ceiling invariant
    if hw.ram_avail_mb >= 6144.0:
        table.add_row(
            "System RAM",
            "[green]OK[/green]",
            f"{hw.ram_avail_mb:.0f} MB available / {hw.ram_total_mb:.0f} MB total (>= 6144 MB ceiling)",
        )
    elif hw.ram_avail_mb >= 2048.0:
        table.add_row(
            "System RAM",
            "[yellow]WARN[/yellow]",
            f"{hw.ram_avail_mb:.0f} MB available (below 6144 MB ceiling; models clamped to available RAM)",
        )
    else:
        table.add_row(
            "System RAM",
            "[red]CRITICAL[/red]",
            f"{hw.ram_avail_mb:.0f} MB available (< 2048 MB; insufficient for local LLMs)",
        )

    # 5. GPU detection
    if hw.gpus:
        gpu_summaries = [
            f"[{g.index}] {g.name} ({g.vram_total_mb:.0f} MB VRAM, {g.vendor.upper()})"
            for g in hw.gpus
        ]
        table.add_row("GPU Hardware", "[green]DETECTED[/green]", "; ".join(gpu_summaries))
    else:
        table.add_row("GPU Hardware", "[yellow]NONE[/yellow]", f"No discrete GPU detected (backend: {hw.backend_target})")

    # 6. Disk space
    if hw.disk_free_gb >= 20.0:
        table.add_row("Disk Space", "[green]OK[/green]", f"{hw.disk_free_gb:.1f} GB free")
    elif hw.disk_free_gb >= 10.0:
        table.add_row("Disk Space", "[yellow]ADEQUATE[/yellow]", f"{hw.disk_free_gb:.1f} GB free (adequate for thin bundles)")
    else:
        table.add_row("Disk Space", "[red]LOW[/red]", f"{hw.disk_free_gb:.1f} GB free (< 10 GB)")

    # 7. Agent Companion packages
    try:
        companion_path = resolve_companion_path()
        pkgs_dir = companion_path / "packages"
        if pkgs_dir.exists():
            pkg_names = [p.name for p in pkgs_dir.iterdir() if p.is_dir()]
            table.add_row(
                "Agent Companion",
                "[green]LOCATED[/green]",
                f"{companion_path} ({len(pkg_names)} packages: {', '.join(pkg_names)})",
            )
        else:
            table.add_row("Agent Companion", "[yellow]FOUND[/yellow]", f"{companion_path}")
    except Exception:
        table.add_row("Agent Companion", "[dim]NOT FOUND[/dim]", "Workspace not found (use --companion-path for packaging)")

    console.print(table)


@app.command("profile")
def profile(
    json_output: bool = typer.Option(
        False, "--json", help="Output hardware profile as raw JSON."
    ),
) -> None:
    """Probe host hardware and display system profile."""
    hardware = detect_hardware()
    if json_output:
        typer.echo(hardware.model_dump_json(indent=2))
        return

    console = Console()
    table = Table(
        title="[bold green]Host Hardware Profile[/bold green]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Operating System", hardware.os_name)
    table.add_row("CPU Brand & Arch", f"{hardware.cpu_brand} ({hardware.cpu_arch})")
    table.add_row("Cores", f"{hardware.physical_cores} Physical / {hardware.logical_cores} Logical")
    table.add_row(
        "SIMD Acceleration",
        f"AVX-512: {hardware.has_avx512}, AVX2: {hardware.has_avx2}, NEON: {hardware.has_neon}",
    )
    table.add_row(
        "System RAM",
        f"{hardware.ram_avail_mb:.0f} MB Available / {hardware.ram_total_mb:.0f} MB Total",
    )
    table.add_row("Free Disk Space", f"{hardware.disk_free_gb:.1f} GB")
    table.add_row("Memory Bandwidth", f"{hardware.bandwidth_gb_s:.1f} GB/s")
    table.add_row("Selected Backend Target", f"[bold green]{hardware.backend_target}[/bold green]")

    if hardware.gpus:
        for gpu in hardware.gpus:
            table.add_row(
                f"GPU [{gpu.index}]",
                f"{gpu.name} ({gpu.vendor.upper()}) | VRAM: {gpu.vram_free_mb:.0f} MB free / {gpu.vram_total_mb:.0f} MB total",
            )
    else:
        table.add_row("GPU Hardware", "[yellow]None detected (CPU fallback backend)[/yellow]")

    console.print(table)


def _run_analysis(
    task: Optional[str] = None,
    priority: Optional[str] = None,
    context_len: Optional[int] = None,
    license: Optional[str] = None,
    interactive: bool = False,
    non_interactive: bool = False,
    offline: bool = False,
    output_md: Optional[Path] = None,
    hf_token: Optional[str] = None,
    sparsity_speedup: float = 0.0,
    live: bool = True,
    search: Optional[str] = None,
    limit: int = 10,
) -> List[Scorecard]:
    console = Console()
    is_interactive = interactive and not non_interactive

    req_permissive = None
    if license:
        req_permissive = license.lower().strip() == "permissive"

    hardware = detect_hardware()
    needs = ask_user_needs(
        interactive=is_interactive,
        task=task,
        priority=priority,
        min_context_tokens=context_len,
        require_permissive_license=req_permissive,
    )

    candidates = get_model_candidates(
        offline=offline,
        hf_token=hf_token,
        live_search=(live and not offline),
        search_query=search,
        task=needs.task,
        limit=limit,
    )
    scorecards = rank_candidates(candidates, hardware, needs, sparsity_speedup=sparsity_speedup)

    display_scorecard_table(scorecards, console=console)

    if scorecards:
        top_sc = scorecards[0]
        card = generate_citation_card(top_sc.candidate, top_sc.evaluation)
        console.print(Panel(card, title="[bold green]Top Recommended Model[/bold green]", border_style="cyan"))

    if output_md:
        output_md = Path(output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        md_text = generate_recommendations_markdown(scorecards, hardware, needs)
        output_md.write_text(md_text, encoding="utf-8")
        console.print(f"[bold green]Saved recommendations to: {output_md}[/bold green]")

    return scorecards


@app.command("analyze")
def analyze(
    task: Optional[str] = typer.Option(
        None, "--task", "-t", help="Target task domain: coding, reasoning, or chat."
    ),
    priority: Optional[str] = typer.Option(
        None, "--priority", "-p", help="Optimization priority: quality, balanced, or speed."
    ),
    context_len: Optional[int] = typer.Option(
        None, "--context-len", "-c", help="Required context window length in tokens."
    ),
    license: Optional[str] = typer.Option(
        None, "--license", "-l", help="License requirement: permissive or any."
    ),
    interactive: bool = typer.Option(
        False, "--interactive/--headless", help="Run interactive questionnaire or use headless arguments."
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Force headless questionnaire execution."
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Use airgapped benchmarks cache without contacting Hugging Face."
    ),
    output_md: Optional[Path] = typer.Option(
        None, "--output-md", "-o", help="Optional path to write RECOMMENDATIONS.md report."
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", help="Optional Hugging Face API token."
    ),
    sparsity_speedup: float = typer.Option(
        0.0, "--sparsity-speedup", help="Neuromorphic ternary sparsity speedup ratio."
    ),
    live: bool = typer.Option(
        True, "--live/--no-live", help="Query live Hugging Face Hub for trending models."
    ),
    search: Optional[str] = typer.Option(
        None, "--search", "-s", help="Custom search query for Hugging Face Hub discovery."
    ),
    limit: int = typer.Option(
        10, "--limit", help="Maximum number of live models to discover."
    ),
) -> List[Scorecard]:
    """Analyze host hardware, interview user needs, and rank candidate models."""
    return _run_analysis(
        task=task,
        priority=priority,
        context_len=context_len,
        license=license,
        interactive=interactive,
        non_interactive=non_interactive,
        offline=offline,
        output_md=output_md,
        hf_token=hf_token,
        sparsity_speedup=sparsity_speedup,
        live=live,
        search=search,
        limit=limit,
    )


@app.command("recommend")
def recommend(
    task: Optional[str] = typer.Option(
        None, "--task", "-t", help="Target task domain: coding, reasoning, or chat."
    ),
    priority: Optional[str] = typer.Option(
        None, "--priority", "-p", help="Optimization priority: quality, balanced, or speed."
    ),
    context_len: Optional[int] = typer.Option(
        None, "--context-len", "-c", help="Context window token limit."
    ),
    license: Optional[str] = typer.Option(
        None, "--license", "-l", help="License filter: permissive or any."
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Offline airgapped mode."
    ),
    non_interactive: bool = typer.Option(
        True, "--non-interactive", help="Non-interactive execution."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Directory to save RECOMMENDATIONS.md."
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", help="Hugging Face API token."
    ),
    live: bool = typer.Option(
        True, "--live/--no-live", help="Query live Hugging Face Hub for trending models."
    ),
    search: Optional[str] = typer.Option(
        None, "--search", "-s", help="Custom search query for Hugging Face Hub discovery."
    ),
    limit: int = typer.Option(
        10, "--limit", help="Maximum number of live models to discover."
    ),
) -> List[Scorecard]:
    """Run needs interview and output ranked recommendations with citations."""
    out_md = (output_dir / "RECOMMENDATIONS.md") if output_dir else None
    return _run_analysis(
        task=task,
        priority=priority,
        context_len=context_len,
        license=license,
        interactive=not non_interactive,
        non_interactive=non_interactive,
        offline=offline,
        output_md=out_md,
        hf_token=hf_token,
        sparsity_speedup=0.0,
        live=live,
        search=search,
        limit=limit,
    )


@app.command("download")
def download(
    repo_id_arg: Optional[str] = typer.Argument(
        None, help="Hugging Face repo ID (e.g. bartowski/Qwen2.5-Coder-7B-Instruct-GGUF)."
    ),
    filename_arg: Optional[str] = typer.Argument(
        None, help="Model GGUF filename in repository."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Hugging Face repo ID."
    ),
    file: Optional[str] = typer.Option(
        None, "--file", "-f", help="Model GGUF filename."
    ),
    dest: Path = typer.Option(
        Path("models"), "--dest", "-d", help="Destination directory."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Alias for --dest."
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Verify offline cache without downloading."
    ),
    expected_sha256: Optional[str] = typer.Option(
        None, "--expected-sha256", help="Expected SHA-256 hash for verification."
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", help="Hugging Face API token."
    ),
) -> Path:
    """Download a model checkpoint with HTTP range resume and SHA-256 verification."""
    console = Console()
    target_repo = repo_id_arg or model
    target_file = filename_arg or file

    if not target_repo or not target_file:
        console.print("[red]Error: Both repository ID and filename must be provided.[/red]")
        raise typer.BadParameter("Both repository ID and filename must be provided.")

    target_dest_dir = output_dir or dest
    target_dest_dir.mkdir(parents=True, exist_ok=True)
    target_file_path = target_dest_dir / target_file

    if offline:
        if target_file_path.exists():
            if expected_sha256:
                actual = compute_sha256(target_file_path)
                if actual != clean_hash(expected_sha256):
                    raise typer.BadParameter(
                        f"SHA-256 mismatch for {target_file_path}: expected {expected_sha256}, got {actual}"
                    )
            console.print(f"[green]Verified offline model file: {target_file_path}[/green]")
            return target_file_path
        else:
            raise typer.BadParameter(
                f"Model file {target_file} not found offline in {target_dest_dir}"
            )

    url = f"https://huggingface.co/{target_repo}/resolve/main/{target_file}"
    console.print(f"[bold blue]Downloading {target_file} from {target_repo}...[/bold blue]")
    result_path = download_file(
        url,
        target_file_path,
        expected_sha256=expected_sha256,
        show_progress=True,
    )
    console.print(f"[bold green]Download complete: {result_path}[/bold green]")
    return result_path


@app.command("build")
def build(
    mode: str = typer.Option(
        "thin",
        "--mode",
        "-m",
        help="Build mode: 'thin' (model downloaded on first run) or 'fat' (model bundled).",
    ),
    format: str = typer.Option(
        "installer",
        "--format",
        help="Installer format: 'installer', 'portable', or 'sfx'.",
    ),
    output_dir: Path = typer.Option(
        Path("dist"),
        "--output-dir",
        "-o",
        help="Output directory for build artifacts.",
    ),
    companion_path: Optional[Path] = typer.Option(
        None,
        "--companion-path",
        help="Path to Agent Companion workspace root.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Hugging Face repo ID or model name for fat installer.",
    ),
    file: Optional[str] = typer.Option(
        None,
        "--file",
        help="Model filename for fat installer.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Airgapped build mode using cached wheels and standalone Python runtime.",
    ),
    wheels_dir: Optional[Path] = typer.Option(
        None,
        "--wheels-dir",
        help="Path to pre-cached wheels directory.",
    ),
    python_dist: Optional[Path] = typer.Option(
        None,
        "--python-dist",
        help="Path to pre-downloaded python-build-standalone archive.",
    ),
    app_name: str = typer.Option(
        "Agent Companion",
        "--app-name",
        help="Application display name.",
    ),
) -> Path:
    """Build a distributable installer package (Thin or Fat mode)."""
    console = Console()
    mode_clean = mode.lower().strip()
    if mode_clean not in ("thin", "fat"):
        raise typer.BadParameter("Mode must be either 'thin' or 'fat'")

    fmt_clean = format.lower().strip()
    if fmt_clean not in ("installer", "portable", "sfx"):
        raise typer.BadParameter("Format must be 'installer', 'portable', or 'sfx'")

    resolved_companion = resolve_companion_path(companion_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hardware = detect_hardware()

    # Pre-flight disk space check
    estimated_size = 8 * 1024 * 1024 * 1024 if mode_clean == "fat" else 500 * 1024 * 1024
    has_space, free_gb, req_gb = check_disk_space_preflight(
        output_dir, required_model_bytes=estimated_size, is_installer_build=True
    )
    if not has_space:
        raise DiskSpaceError(
            f"Insufficient disk space in {output_dir}: {free_gb:.1f} GB free, {req_gb:.1f} GB required."
        )

    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    wheels_dest = bundle_dir / "wheels"
    wheels_dest.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Building monorepo wheels from {resolved_companion}...[/bold blue]")
    monorepo_wheels = stage_and_build_monorepo_wheels(resolved_companion, wheels_dest)
    packager_wheel = build_packager_wheel(wheels_dest)
    console.print(f"[green]Built {len(monorepo_wheels) + 1} workspace wheels.[/green]")

    console.print("[bold blue]Vendoring third-party dependencies...[/bold blue]")
    reqs = extract_and_filter_requirements(resolved_companion)
    target_os = (
        "windows"
        if sys.platform == "win32"
        else ("darwin" if sys.platform == "darwin" else "linux")
    )
    third_party_wheels = vendor_third_party_wheels(
        reqs,
        wheels_dest,
        target_os=target_os,
        wheels_cache_dir=wheels_dir,
        offline=offline,
    )
    console.print(f"[green]Vendored {len(third_party_wheels)} third-party wheels.[/green]")

    console.print("[bold blue]Acquiring pinned Python standalone runtime...[/bold blue]")
    runtime_dest = bundle_dir / "runtime"
    runtime_dest.mkdir(parents=True, exist_ok=True)
    python_bin = acquire_python_standalone(
        runtime_dest,
        python_dist_path=python_dist,
        target_os=target_os,
        offline=offline,
    )
    console.print(f"[green]Acquired Python runtime at {python_bin}.[/green]")

    # Resolve candidate model
    candidates = get_model_candidates(offline=True)
    selected_model: CandidateModel
    if file or model:
        matching = [
            c for c in candidates
            if (file and c.filename == file) or (model and c.repo_id == model)
        ]
        if matching:
            selected_model = matching[0]
        else:
            selected_model = CandidateModel(
                repo_id=model or "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
                filename=file or "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
                quant_method="Q4_K_M",
                param_count_billions=7.61,
                size_mb=4680.0,
            )
    else:
        needs = UserNeeds(task="coding", priority="balanced")
        scorecards = rank_candidates(candidates, hardware, needs)
        selected_model = scorecards[0].candidate if scorecards else candidates[0]

    # Staging model for fat mode
    if mode_clean == "fat":
        models_dest = bundle_dir / "models"
        models_dest.mkdir(parents=True, exist_ok=True)
        dest_model_file = models_dest / selected_model.filename

        if not dest_model_file.exists():
            local_src_candidates = [
                Path("models") / selected_model.filename,
                Path(selected_model.filename),
            ]
            found_local = None
            for lc in local_src_candidates:
                if lc.exists():
                    found_local = lc
                    break

            if found_local:
                shutil.copy2(found_local, dest_model_file)
            elif offline:
                raise RuntimeError(
                    f"Offline fat build failed: Model {selected_model.filename} not found locally."
                )
            else:
                url = f"https://huggingface.co/{selected_model.repo_id}/resolve/main/{selected_model.filename}"
                download_file(url, dest_model_file, show_progress=True)

    console.print("[bold blue]Injecting Agent Companion runtime configuration...[/bold blue]")
    inject_agent_companion_config(bundle_dir, selected_model, hardware)

    console.print(f"[bold blue]Packaging installer into format '{fmt_clean}'...[/bold blue]")
    final_artifact: Path
    if fmt_clean == "sfx" or (fmt_clean == "installer" and sys.platform != "win32"):
        sfx_out = output_dir / f"{app_name.replace(' ', '')}-Installer.run"
        final_artifact = build_posix_sfx(bundle_dir, sfx_out, app_name=app_name)
    else:
        final_artifact = build_windows_installer(bundle_dir, output_dir, app_name=app_name)

    console.print(
        Panel(
            f"[bold green]Artifact Created:[/bold green] {final_artifact}\n"
            f"[bold cyan]Mode:[/bold cyan] {mode_clean.upper()}\n"
            f"[bold cyan]Model:[/bold cyan] {selected_model.filename}\n"
            f"[bold cyan]Size:[/bold cyan] {final_artifact.stat().st_size / (1024 * 1024):.1f} MB",
            title="[bold green]Build Succeeded[/bold green]",
            border_style="green",
        )
    )
    return final_artifact


@app.command("install")
def install(
    bundle_path: Optional[Path] = typer.Option(
        None, "--bundle-path", "-b", help="Path to bundled folder or archive."
    ),
    target_dir: Optional[Path] = typer.Option(
        None, "--target-dir", "-t", "--install-dir", help="Target installation directory."
    ),
    model_path: Optional[Path] = typer.Option(
        None, "--model-path", help="Optional path to model file to copy."
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Run without prompts."
    ),
) -> Path:
    """Execute target-machine installer setup and offline wheel installation."""
    console = Console()

    # Determine target directory
    if target_dir:
        install_dir = Path(target_dir).resolve()
    else:
        if sys.platform == "win32":
            localappdata = os.environ.get(
                "LOCALAPPDATA", str(Path.home() / "AppData" / "Local")
            )
            install_dir = Path(localappdata) / "Programs" / "Agent Companion"
        else:
            install_dir = Path.home() / ".local" / "share" / "agent-companion"

    install_dir.mkdir(parents=True, exist_ok=True)

    # Determine bundle directory
    if bundle_path:
        src_bundle = Path(bundle_path).resolve()
        if src_bundle.is_file() and (
            src_bundle.suffix.lower() == ".zip" or zipfile.is_zipfile(src_bundle)
        ):
            with zipfile.ZipFile(src_bundle, "r") as z:
                z.extractall(install_dir)
        elif src_bundle.is_dir() and src_bundle != install_dir:
            for item in src_bundle.iterdir():
                dst = install_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dst)
    else:
        cwd = Path.cwd().resolve()
        if (cwd / "wheels").exists() or (cwd / "runtime").exists():
            if cwd != install_dir:
                for item in cwd.iterdir():
                    dst = install_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dst)
        elif (Path("dist") / "bundle").exists():
            dist_bundle = (Path("dist") / "bundle").resolve()
            if dist_bundle != install_dir:
                for item in dist_bundle.iterdir():
                    dst = install_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dst)

    # Locate Python binary
    python_candidates = [
        install_dir / "runtime" / "python.exe",
        install_dir / "runtime" / "bin" / "python3",
        install_dir / "runtime" / "bin" / "python",
        Path.cwd() / "runtime" / "python.exe",
        Path.cwd() / "runtime" / "bin" / "python3",
    ]
    python_bin = Path(sys.executable)
    for cand in python_candidates:
        if cand.exists():
            python_bin = cand
            break

    # Locate wheels
    wheels_dir = install_dir / "wheels"
    if not wheels_dir.exists() and (Path.cwd() / "wheels").exists():
        wheels_dir = Path.cwd() / "wheels"

    if wheels_dir.exists():
        console.print(f"[bold blue]Installing runtime dependencies from {wheels_dir}...[/bold blue]")
        cmd = [
            str(python_bin),
            "-m",
            "pip",
            "install",
            "--no-index",
            f"--find-links={wheels_dir}",
            "shared_contracts",
            "model_runtime",
            "security_server",
            "document_pipeline",
            "agent_tooling",
            "companion_packager",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            console.print(f"[yellow]Warning during offline pip install: {res.stderr or res.stdout}[/yellow]")
        else:
            console.print("[green]Installed runtime wheels successfully.[/green]")

    # Optional model staging
    if model_path and Path(model_path).exists():
        models_dir = install_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, models_dir / Path(model_path).name)

    # Create shortcut fail-soft
    create_desktop_shortcut(install_dir)

    console.print(f"[bold green]Agent Companion installed successfully at: {install_dir}[/bold green]")
    return install_dir
