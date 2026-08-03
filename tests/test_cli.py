from homebench.cli import _inject_default_command, build_parser


def test_inject_default_command():
    assert _inject_default_command([]) == ["run"]
    assert _inject_default_command(["--no-tui"]) == ["run", "--no-tui"]
    assert _inject_default_command(["-m", "x"]) == ["run", "-m", "x"]
    # explicit subcommands are left untouched
    assert _inject_default_command(["list"]) == ["list"]
    assert _inject_default_command(["run", "--limit", "2"]) == ["run", "--limit", "2"]
    # global help/version bypass the default
    assert _inject_default_command(["--version"]) == ["--version"]


def test_provider_flag_reaches_list_command():
    parser = build_parser()
    args = parser.parse_args(["list", "--provider", "lmstudio"])
    assert args.command == "list"
    assert args.provider == "lmstudio"


def test_run_flags_parse():
    parser = build_parser()
    args = parser.parse_args(_inject_default_command(
        ["--provider", "lmstudio", "--no-tui", "--limit", "3", "--judge", "m"]
    ))
    assert args.command == "run"
    assert args.provider == "lmstudio"
    assert args.no_tui is True
    assert args.limit == 3
    assert args.judge == "m"
